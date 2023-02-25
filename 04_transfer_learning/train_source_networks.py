import os

import numpy as np
import pandas as pd
import torch
import wandb
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from delphi.utils.tools import ToTensor, compute_accuracy, convert_wandb_config
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


def set_random_seed(seed):
    import random
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()  # can be used in pytorch dataloaders for reproducible sample selection when shuffle=True
    g.manual_seed(seed)

    return g


g = set_random_seed(2020)


def wandb_plots(y_true, y_pred, y_prob, class_labels, dataset):
    wandb.log({
        f"{dataset}-ROC": wandb.plot.roc_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                               title=f"{dataset}-ROC"),
        f"{dataset}-PR": wandb.plot.pr_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                             title=f"{dataset}-PR"),
        f"{dataset}-ConfMat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=class_labels,
                                                          title=f"{dataset}-ConfMat")
    })


def wandb_plots(y_true, y_pred, y_prob, class_labels, dataset):
    wandb.log({
        f"{dataset}-ROC": wandb.plot.roc_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                               title=f"{dataset}-ROC"),
        f"{dataset}-PR": wandb.plot.pr_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                             title=f"{dataset}-PR"),
        f"{dataset}-ConfMat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=class_labels,
                                                          title=f"{dataset}-ConfMat")
    })


def train_net(model, config, save_name, data_train, data_valid, data_test, logwandb=True):
    dl_test = DataLoader(data_test, batch_size=config.batch_size, shuffle=True, generator=g)
    dl_train = DataLoader(data_train, batch_size=config.batch_size, shuffle=True, generator=g)
    dl_valid = DataLoader(data_valid, batch_size=config.batch_size, shuffle=True, generator=g)

    best_loss, best_acc = 100, 0
    loss_acc = []
    train_stats, valid_stats = [], []
    patience = 9
    patience_ctr = 0

    # loop for the above set number of epochs
    for epoch in range(0, config.epochs):
        _, _ = model.fit(dl_train, lr=config.learning_rate, device=DEVICE)

        # for validating or testing set the network into evaluation mode such that layers like dropout are not active
        with torch.no_grad():
            tloss, tstats = model.fit(dl_train, device=DEVICE, train=False)
            vloss, vstats = model.fit(dl_valid, device=DEVICE, train=False)

        tacc = compute_accuracy(tstats[:, -2], tstats[:, -1])
        vacc = compute_accuracy(vstats[:, -2], vstats[:, -1])

        loss_acc.append(pd.DataFrame([[tloss, vloss, tacc, vacc]],
                                     columns=["train_loss", "valid_loss", "train_acc", "valid_acc"]))

        train_stats.append(pd.DataFrame(tstats.tolist(), columns=[*model.config["classes"], *["real", "predicted"]]))
        train_stats[epoch]["epoch"] = epoch
        valid_stats.append(pd.DataFrame(vstats.tolist(), columns=[*model.config["classes"], *["real", "predicted"]]))
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
            wandb_plots(vstats[:, -2], vstats[:, -1], vstats[:, :-2], model.config["classes"], "valid")

            # reset the patience counter
            patience_ctr = 0

        else:
            patience_ctr += 1

        if patience_ctr > patience:
            print('Reached patience. Stopping training and continuing with test set.')
            break

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
    testacc = compute_accuracy(teststats[:, -2], teststats[:, -1])
    wandb.run.summary["test_accuracy"] = testacc

    wandb.log({"test_accuracy": testacc, "test_loss": testloss})
    wandb_plots(teststats[:, -2], teststats[:, -1], teststats[:, :-2], model.config["classes"], "test")

    wandb.finish()


# define the training function with the wandb init
def main():
    # here we initialize weights&biases.
    with wandb.init() as run:
        tmp = [value for key, value in classes_of_ds.items() if key not in {run.config.left_out_task}]
        class_labels = [j for val in tmp for j in val]

        converted_config = convert_wandb_config(wandb.config, BrainStateClassifier3d._REQUIRED_PARAMS)
        converted_config["classes"] = class_labels

        # the zero means to take all samples found in a directory
        data_test = NiftiDataset("../t-maps/test", class_labels, 0, device=DEVICE, transform=ToTensor())

        # we will split the train dataset into a train (80%) and validation (20%) set.
        data_train_full = NiftiDataset("../t-maps/train", class_labels, 0, device=DEVICE, transform=ToTensor())

        # we want one stratified shuffled split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2020)
        idx_train, idx_valid = next(sss.split(data_train_full.data, data_train_full.labels))

        data_train = torch.utils.data.Subset(data_train_full, idx_train)
        data_valid = torch.utils.data.Subset(data_train_full, idx_valid)

        model = BrainStateClassifier3d((91, 109, 91), len(class_labels), converted_config)
        model.to(DEVICE)

        save_name = os.path.join("models", f"hcp-without-{run.config.left_out_task}")
        wandb.run.name = f"hcp-without-{run.config.left_out_task}"

        # now train the netwok
        train_net(model, wandb.config, save_name, data_train, data_valid, data_test)



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
    main()
