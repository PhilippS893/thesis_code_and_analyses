import argparse
import os

import delphi.utils.tools as tools
import pandas as pd
import torch
import wandb
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import DataLoader

from utils.random import set_random_seed
from utils.wandb_funcs import wandb_plots

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.environ['WANDB_MODE'] = 'offline'


class_labels = ["footleft", "footright", "handleft", "handright", "tongue", "rest_MOTOR",
                "body", "face", "place", "tool", "rest_WM",
                "match", "relation", "rest_RELATIONAL",
                "mental", "rnd", "rest_SOCIAL"]
"""
class_labels = ["footleft", "footright", "handleft", "handright", "tongue",
                "body", "face", "place", "tool",
                "match", "relation",
                "mental", "rnd"]
"""
# class_labels = ["footleft", "footright", "handleft", "handright", "tongue", "rest_MOTOR"]
#class_labels = ["body", "face", "place", "tool", "rest_WM"]


def train_network(config=None, num_folds=8, run_splits=1) -> None:
    if config is None:
        config = vars(get_argparse().parse_args())

    hps = tools.read_config(config["hyperparameter"])

    data_train_full = NiftiDataset("../v-maps/train", class_labels, 0, device=DEVICE,
                                   transform=tools.ToTensor(), shuffle_labels=config["shuffle_labels"])

    data_test = NiftiDataset("../v-maps/test", class_labels, 0, device=DEVICE,
                             transform=tools.ToTensor())

    sss = StratifiedKFold(n_splits=num_folds)
    splits = sss.split(data_train_full.data, data_train_full.labels)

    for fold in range(run_splits):

        (idx_train, idx_valid) = next(splits)

        this_seed = config["seed"] + fold
        g = set_random_seed(this_seed)

        job_type = f"{config['job_type']}-shuffled" if config["shuffle_labels"] else f"{config['job_type']}-real"
        run_name = f"seed-{config['seed']}_fold-{fold:02d}"

        wandb_kwargs = {
            "entity": "philis893",
            "project": "thesis",
            "group": config["wandb_group"],  # "transfer-learning-diffparams"
            "name": run_name,
            "job_type": job_type,
            "allow_val_change": True,
        }

        save_name = os.path.join("models", job_type, run_name)
        if os.path.exists(save_name):
            continue

        data_train = torch.utils.data.Subset(data_train_full, idx_train)
        data_valid = torch.utils.data.Subset(data_train_full, idx_valid)

        # we now use the wandb context to track the training and evaluation process.
        # all settings and changes will be reset at the beginning of the fold-loop. (see line 11)
        with wandb.init(config=hps, **wandb_kwargs) as run:

            model_cfg = tools.convert_wandb_config(run.config, BrainStateClassifier3d._REQUIRED_PARAMS)
            # setup a model with the parameters given in model_cfg
            model = BrainStateClassifier3d(config["input_dims"], len(class_labels), model_cfg)
            model.config["class_labels"] = class_labels

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
            description='test parser',
        )

    parser.add_argument(
        '--hyperparameter',
        metavar='.yaml',
        default='best_hp.yaml',
        type=str,
        help='a path to a .yaml file containing hyperparameter configs (default: hyperparameter.yaml)'
    )

    parser.add_argument(
        '--seed',
        metavar='INT',
        default=2020,
        type=int,
        help="set a random seed for reproducibility (default: 2020)"
    )

    parser.add_argument(
        '--shuffle_labels',
        metavar='BOOL',
        default=False,
        type=bool,
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
        '--job_type',
        metavar='STR',
        default="multi-w-rest",
        type=str,
        help="the name of the job (important to group in wandb)"
    )

    parser.add_argument(
        '--wandb_group',
        metavar='STR',
        default="volumes",
        type=str,
        help="group name of the current runs. Important to group in wandb"
    )

    return parser


if __name__ == '__main__':
    train_network()
