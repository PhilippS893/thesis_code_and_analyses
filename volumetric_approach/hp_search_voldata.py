import os
import time

import pandas as pd
import torch
import wandb
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from delphi.utils.tools import convert_wandb_config, ToTensor, compute_accuracy
from torch.utils.data import DataLoader

from utils.random import set_random_seed
from utils.wandb_funcs import wandb_plots

# SET SOME WANDB THINGS
# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_ENTITY'] = "philis893"  # this is my wandb account name. This can also be a group name, for example
os.environ['WANDB_PROJECT'] = "thesis"

g = set_random_seed(2020)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# External harddrive directory
data_dir = "/media/philippseidel/5tb/hcp_download/data"


def main():
    # for debugging
    # hp = read_config("default_vals.yaml")

    # class_labels = sorted([os.path.split(x)[-1] for x in glob.glob(os.path.join(settings.train_dir, "*"))])
    class_labels = sorted(["handleft", "handright", "footleft", "footright", "tongue"])
    # initialize wandb
    wandb.init(entity=os.environ['WANDB_ENTITY'], project=os.environ['WANDB_PROJECT'],
               group="volume-input", job_type="hp_exploration", allow_val_change=True)
    config = wandb.config

    t_stamp = time.time()
    save_name = os.path.join("models", f"motor-wo-rest-{t_stamp}")
    wandb.run.name = f"motor-wo-rest-{t_stamp}"

    # convert the wandb config into a working format
    cfg = convert_wandb_config(config, BrainStateClassifier3d._REQUIRED_PARAMS)

    # create the model
    model = BrainStateClassifier3d((91, 109, 91), len(class_labels), cfg)
    model.float().to(DEVICE)
    wandb.watch(model, log="all")

    # update the wandb config
    config.update(model.config, allow_val_change=True)

    # setup dataloaders
    train_loader = DataLoader(
        NiftiDataset(os.path.join(data_dir, "all/train"), class_labels, 2000, device=DEVICE, transform=ToTensor()),
        batch_size=config.batch_size, shuffle=True, generator=g
    )

    valid_loader = DataLoader(
        NiftiDataset(os.path.join(data_dir, "all/valid"), class_labels, 380, device=DEVICE, transform=ToTensor()),
        batch_size=config.batch_size, shuffle=True, generator=g
    )

    test_loader = DataLoader(
        NiftiDataset(os.path.join(data_dir, "all/test"), class_labels, 400, device=DEVICE, transform=ToTensor()),
        batch_size=config.batch_size, shuffle=True, generator=g
    )

    best_loss, best_acc = 100, 0
    loss_acc = []
    train_stats, valid_stats = [], []
    patience = 9
    patience_ctr = 0

    for epoch in range(config.epochs):
        _, _ = model.fit(train_loader, lr=config.learning_rate, device=DEVICE, train=True)

        with torch.no_grad():
            tloss, tstats = model.fit(train_loader, lr=config.learning_rate, device=DEVICE, train=False)
            vloss, vstats = model.fit(valid_loader, lr=config.learning_rate, device=DEVICE, train=False)
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
        })

        print("Epoch: %d, train-loss: %2.3f, train-acc: %1.2f, valid-loss: %2.3f, valid-acc: %1.2f" %
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
        testloss, teststats = model.fit(test_loader, train=False)
    testacc = compute_accuracy(teststats[:, -2], teststats[:, -1])
    wandb.run.summary["test_accuracy"] = testacc

    wandb.log({"test_accuracy": testacc, "test_loss": testloss})
    wandb_plots(teststats[:, -2], teststats[:, -1], teststats[:, :-2], class_labels, "test")

    wandb.finish()


if __name__ == '__main__':
    main()
