import argparse
import os

import delphi.utils.tools as tools
import pandas as pd
import torch
import wandb
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchinfo import summary

from utils.random import set_random_seed
from utils.wandb_funcs import wandb_plots

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


def train_network(config=None, num_folds=10) -> None:
    if config is None:
        config = vars(get_argparse().parse_args())

    if config["transfer_learning"]:
        path_to_pretrained = os.path.join(
            "models", "source-nets2",
            f"hcp-without-{config['task'].lower()}_seed-7474_fold-{config['best_fold']:02d}"
        )
        hps = tools.read_config(os.path.join(path_to_pretrained, "config.yaml"))
    else:
        hps = tools.read_config(config["hyperparameter"])

    class_labels = classes_of_ds[config["task"]]

    data_train_full = NiftiDataset("../t-maps/train", class_labels, 0, device=DEVICE, transform=tools.ToTensor())
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
        run_name = f"{config['task'].lower()}_samplesize-{config['sample_size']}"

        wandb_kwargs = {
            "entity": "philis893",
            "project": "thesis",
            "group": config["wandb_group"],
            "name": f"seed-{config['seed']}_fold-{fold:02d}",
            "job_type": run_name,
            "allow_val_change": True,
        }

        save_name = os.path.join("models", job_type, f"{run_name}_seed-{config['seed']}_fold-{fold:02d}")
        if os.path.exists(save_name):
            continue

        data_train = torch.utils.data.Subset(data_train_full, idx_train)
        data_valid = torch.utils.data.Subset(data_train_full, idx_valid)

        # we now use the wandb context to track the training and evaluation process.
        # all settings and changes will be reset at the beginning of the fold-loop.
        with wandb.init(config=hps, **wandb_kwargs) as run:

            if config["transfer_learning"]:
                model = BrainStateClassifier3d(path_to_pretrained)
                model.out = torch.nn.Linear(model.out.in_features, out_features=len(class_labels))
                model.config["n_classes"] = len(class_labels)
                model.config["class_labels"] = class_labels
            else:
                # please note that this conversion is unnecessary if not using w&b!
                model_cfg = tools.convert_wandb_config(run.config, BrainStateClassifier3d._REQUIRED_PARAMS)
                # setup a model with the parameters given in model_cfg
                model = BrainStateClassifier3d(config["input_dims"], len(class_labels), model_cfg)
                model.config["class_labels"] = class_labels

            print(summary(model, (1, 1, 91, 109, 91)))

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
                _, _ = model.fit(dl_train, lr=run.config.learning_rate, device=DEVICE,
                                 **{"weight_decay": run.config.weight_decay})

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

            # load the best performing model
            model = BrainStateClassifier3d(save_name)
            model.to(DEVICE)

            # EVALUATE THE MODEL ON THE TEST DATA
            with torch.no_grad():
                testloss, teststats = model.fit(dl_test, train=False)

            df_test = pd.DataFrame(teststats.tolist(), columns=[*class_labels, *["real", "predicted"]])
            df_test.to_csv(os.path.join(save_name, "test_stats.csv"), index=False)
            testacc = tools.compute_accuracy(teststats[:, -2], teststats[:, -1])
            wandb.run.summary["test_accuracy"] = testacc

            wandb.log({"test_accuracy": testacc, "test_loss": testloss})
            wandb_plots(teststats[:, -2], teststats[:, -1], teststats[:, :-2], class_labels, "test")

            wandb.finish()


def get_argparse(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Parameters for training neural networks either from scratch or using transfer learning',
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

    parser.add_argument(
        '--best_fold',
        metavar='INT',
        default=0,
        type=int,
        help="a number indicating the best fold of the chosen source network (default: 0)"
    )

    parser.add_argument(
        '--wandb_group',
        metavar='STR',
        default="transfer-learning",
        type=str,
        help="group name of the current runs. Important to group in wandb"
    )

    return parser


if __name__ == '__main__':
    train_network()
