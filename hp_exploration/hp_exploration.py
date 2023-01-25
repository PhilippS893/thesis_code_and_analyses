import os
import time

import pandas as pd
import torch
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from delphi.utils.tools import ToTensor, compute_accuracy, convert_wandb_config
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

import wandb
from utils.random import set_random_seed
from utils.wandb_funcs import wandb_plots

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g = set_random_seed(2020)

class_labels = sorted(["handleft", "handright", "footleft", "footright", "tongue"])

data_test = NiftiDataset("../t-maps/test", class_labels, 0, device=DEVICE, transform=ToTensor())

# we will split the train dataset into a train (80%) and validation (20%) set.
data_train_full = NiftiDataset("../t-maps/train", class_labels, 0, device=DEVICE, transform=ToTensor())

# we want one stratified shuffled split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2020)
idx_train, idx_valid = next(sss.split(data_train_full.data, data_train_full.labels))

data_train = torch.utils.data.Subset(data_train_full, idx_train)
data_valid = torch.utils.data.Subset(data_train_full, idx_valid)


def train_net(model, config, save_name, logwandb=True):
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
    wandb_plots(teststats[:, -2], teststats[:, -1], teststats[:, :-2], class_labels, "test")

    wandb.finish()


# define the training function with the wandb init
def main():
    # here we initialize weights&biases.
    with wandb.init() as run:
        # here's the promised conversion of the wandb.config
        # this results into a dict that contains key-value pairs that we can use to configure our network:
        # converted_config['lin_neurons'] = [512, 8, 128]

        converted_config = convert_wandb_config(wandb.config, BrainStateClassifier3d._REQUIRED_PARAMS)

        model = BrainStateClassifier3d((91, 109, 91), len(class_labels), converted_config)
        model.to(DEVICE)

        # We do not necessarily need this line but it is nice to update the config.
        # wandb.config.update(model.config, allow_val_change=True)

        t_stamp = time.time()
        save_name = os.path.join("models", f"motor-explo_{t_stamp}")
        wandb.run.name = f"motor-explo-{t_stamp}"

        # now train the netwok, yay!
        train_net(model, wandb.config, save_name)


if __name__ == '__main__':
    main()
