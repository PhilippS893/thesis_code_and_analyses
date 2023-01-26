import argparse
import os

import delphi.utils.tools as tools
import numpy as np
import pandas as pd
import torch
import wandb
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.environ['WANDB_MODE'] = 'offline'

classes_of_ds = {
    'MOTOR': sorted(['handleft', 'handright', 'footleft', 'footright', 'tongue']),
    'GAMBLING': sorted(['reward', 'loss']),
    'SOCIAL': sorted(['mental', 'random']),
    'WM': sorted(['body', 'face', 'place', 'tool']),
    'RELATIONAL': sorted(['match', 'relation']),
    'EMOTION': sorted(['emotion', 'neut']),
    'LANGUAGE': ['story', 'math'],
}


def wandb_plots(y_true, y_pred, y_prob, class_labels, dataset):
    r"""
    Function to plot receiver-operating-characteristics, precision-recall curves, and confusion matrices
    in the w&b online platform.

    Parameters
    ----------
    y_true: list of true labels
    y_pred: list of predicted labels
    y_prob: probability values of the network prediction
    class_labels: the names of the classes to predict
    dataset: string; name of the current dataset
    """
    wandb.log({
        f"{dataset}-ROC": wandb.plot.roc_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                               title=f"{dataset}-ROC"),
        f"{dataset}-PR": wandb.plot.pr_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                             title=f"{dataset}-PR"),
        f"{dataset}-ConfMat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=class_labels,
                                                          title=f"{dataset}-ConfMat")
    })


def set_random_seed(seed):
    import random
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()  # can be used in pytorch dataloaders for reproducible sample selection when shuffle=True
    g.manual_seed(seed)

    return g


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train_network(config=None, num_folds=10) -> None:
    if config is None:
        config = vars(get_argparse().parse_args())

    if config["transfer_learning"]:
        path_to_pretrained = os.path.join("models", f"hcp-without-{config['task']}")
        hps = tools.read_config(os.path.join(path_to_pretrained, "config.yaml"))
    else:
        hps = tools.read_config(config["hyperparameter"])
    class_labels = classes_of_ds[config["task"]]

    # we will split the train dataset into a train (80%) and validation (20%) set.
    data_train_full = NiftiDataset("../t-maps/train", class_labels, 0, device=DEVICE,
                                   transform=tools.ToTensor())

    data_test = NiftiDataset("../t-maps/test", class_labels, 0, device=DEVICE, transform=tools.ToTensor())

    # we want a stratified shuffled split
    if config["sample_size"] == 0:
        sss = StratifiedShuffleSplit(n_splits=num_folds, test_size=20*len(class_labels), random_state=2020)
    else:
        sss = StratifiedShuffleSplit(n_splits=num_folds, train_size=config["sample_size"]*len(class_labels),
                                     test_size=20*len(class_labels), random_state=2020)

    for fold, (idx_train, idx_valid) in enumerate(sss.split(data_train_full.data, data_train_full.labels)):

        this_seed = config["seed"] + fold
        g = set_random_seed(this_seed)

        job_type = "pretrained" if config["transfer_learning"] else "from-scratch"
        run_name = f"{job_type}_{config['task'].lower()}_samplesize-{config['sample_size']}"

        wandb_kwargs = {
            "entity": "philis893",
            "project": "thesis",
            "group": "transfer-learning",
            "name": f"seed-{config['seed']}_fold-{fold:02d}",
            "job_type": run_name,
            "allow_val_change": True,
        }

        save_name = os.path.join("models", f"{run_name}_seed-{config['seed']}_fold-{fold:02d}")
        if os.path.exists(save_name):
            continue

        data_train = torch.utils.data.Subset(data_train_full, idx_train)
        data_valid = torch.utils.data.Subset(data_train_full, idx_valid)

        # we now use the wandb context to track the training and evaluation process.
        # all settings and changes will be reset at the beginning of the fold-loop. (see line 11)
        with wandb.init(config=hps, **wandb_kwargs) as run:

            if config["transfer_learning"]:
                model = BrainStateClassifier3d(path_to_pretrained)
                model.out = torch.nn.Linear(model.out.in_features, out_features=len(class_labels))
                model.config["n_classes"] = len(class_labels)
                model.config["classes"] = class_labels
                del model.config["left_out_task"]
            else:
                # please note that this conversion is unnecessary if not using w&b!
                model_cfg = tools.convert_wandb_config(run.config, BrainStateClassifier3d._REQUIRED_PARAMS)
                # setup a model with the parameters given in model_cfg
                model = BrainStateClassifier3d(config["input_dims"], len(class_labels), model_cfg)

            run.config.update(config, allow_val_change=True)
            run.config.update(model.config, allow_val_change=True)

            print(model.config)
            model.to(DEVICE);

            dl_train = DataLoader(data_train, batch_size=run.config.batch_size, shuffle=True, generator=g)
            dl_valid = DataLoader(data_valid, batch_size=run.config.batch_size, shuffle=True, generator=g)
            dl_test = DataLoader(data_test, batch_size=run.config.batch_size, shuffle=False, generator=g)

            best_loss, best_acc = 100, 0
            loss_acc = []
            train_stats, valid_stats = [], []

            # loop for the above set number of epochs
            for epoch in range(run.config.epochs):
                _, _ = model.fit(dl_train, lr=run.config.learning_rate, device=DEVICE, **{"weight_decay": 0.0001})

                # for validating or testing set the network into evaluation mode such
                # that layers like dropout are not active
                with torch.no_grad():
                    tloss, tstats = model.fit(dl_train, device=DEVICE, train=False)
                    vloss, vstats = model.fit(dl_valid, device=DEVICE, train=False)

                # the model.fit() method has 2 output parameters: loss, stats = model.fit()
                # the first parameter is simply the loss for each sample
                # the second parameter is a matrix of n_classes+2-by-n_samples
                # the first n_classes columns are the output probabilities of the model per class
                # the second to last column (i.e., [:, -2]) represents the real labels
                # the last column (i.e., [:, -1]) represents the predicted labels
                tacc = tools.compute_accuracy(tstats[:, -2], tstats[:, -1])
                vacc = tools.compute_accuracy(vstats[:, -2], vstats[:, -1])

                loss_acc.append(pd.DataFrame([[tloss, vloss, tacc, vacc]],
                                             columns=["train_loss", "valid_loss", "train_acc", "valid_acc"]))

                train_stats.append(pd.DataFrame(tstats.tolist(), columns=[*class_labels, *["real", "predicted"]]))
                train_stats[epoch]["epoch"] = epoch
                valid_stats.append(pd.DataFrame(vstats.tolist(), columns=[*class_labels, *["real", "predicted"]]))
                valid_stats[epoch]["epoch"] = epoch

                wandb.log({
                    "train_acc": tacc, "train_loss": tloss,
                    "valid_acc": vacc, "valid_loss": vloss
                }, step=epoch)

                print('Epoch=%03d, train_loss=%2.3f, train_acc=%1.3f, valid_loss=%2.3f, valid_acc=%1.3f' %
                      (epoch, tloss, tacc, vloss, vacc))

                if (vacc >= best_acc) and (vloss <= best_loss):
                    # assign the new best values
                    best_acc, best_loss = vacc, vloss
                    wandb.run.summary["best_valid_accuracy"] = best_acc
                    wandb.run.summary["best_valid_epoch"] = epoch
                    # save the current best model
                    model.save(save_name)
                    # plot some graphs for the validation data
                    wandb_plots(vstats[:, -2], vstats[:, -1], vstats[:, :-2], class_labels, "valid")

            # save the files
            full_df = pd.concat(loss_acc)
            full_df.to_csv(os.path.join(save_name, "loss_acc_curves.csv"), index=False)
            full_df = pd.concat(train_stats)
            full_df.to_csv(os.path.join(save_name, "train_stats.csv"), index=False)
            full_df = pd.concat(valid_stats)
            full_df.to_csv(os.path.join(save_name, "valid_stats.csv"), index=False)

            # EVALUATE THE MODEL ON THE TEST DATA
            with torch.no_grad():
                testloss, teststats = model.fit(dl_test, train=False)
            testacc = tools.compute_accuracy(teststats[:, -2], teststats[:, -1])
            wandb.run.summary["test_accuracy"] = testacc

            wandb.log({"test_accuracy": testacc, "test_loss": testloss})
            wandb_plots(teststats[:, -2], teststats[:, -1], teststats[:, :-2], class_labels, "test")

            wandb.finish()


def get_argparse(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description='test parser',
        )

    parser.add_argument(
        '--task',
        metavar='TASK',
        default='MOTOR',
        type=str,
        help='name of task (default: MOTOR)'
    )

    parser.add_argument(
        '--sample_size',
        metavar='INT',
        default='0',
        type=int,
        help='number of samples to consider (default: 0; all samples in directory)'
    )

    parser.add_argument(
        '--hyperparameter',
        metavar='.yaml',
        default='hyperparameter.yaml',
        type=str,
        help='a path to a .yaml file containing hyperparameter configs (default: hyperparameter.yaml)'
    )

    parser.add_argument(
        '--transfer_learning',
        metavar='BOOLEAN',
        default=False,
        type=bool,
        help="check if transfer learning is supposed to be used (default: False)"
    )

    parser.add_argument(
        '--seed',
        metavar='INT',
        default=2020,
        type=int,
        help="set a random seed for reproducibility (default: 2020)"
    )

    parser.add_argument(
        '--input_dims',
        metavar='TUPLE',
        default=(91, 109, 91),
        type=tuple,
        help="a tuple indicating the dimensions of your input data (default: (91, 109, 91); the MNI brain dimensions)"
    )

    return parser


if __name__ == '__main__':
    train_network()
