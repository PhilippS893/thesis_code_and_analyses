import os

import pandas as pd
import torch
import wandb
from delphi.networks.ConvNets import BrainStateClassifier3d
from delphi.utils.datasets import NiftiDataset
from delphi.utils.tools import ToTensor, compute_accuracy, read_config
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from utils.random import set_random_seed
from utils.wandb_funcs import wandb_plots, reset_wandb_env

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g = set_random_seed(2020)

# set the wandb sweep config
# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = "/media/philippseidel/5tb/thesis_code_and_analyses/01_train_classifier"


def main(num_folds=10, shuffle_labels=False):
    class_labels = sorted(["handleft", "handright", "footleft", "footright", "tongue"])
    print(class_labels)

    data_test = NiftiDataset("../t-maps/test", class_labels, 0, device=DEVICE, transform=ToTensor())

    # we will split the train dataset into a train (80%) and validation (20%) set.
    data_train_full = NiftiDataset("../t-maps/train", class_labels, 0, device=DEVICE, transform=ToTensor(),
                                   shuffle_labels=shuffle_labels)

    # hp = read_config("hyperparameter.yaml")
    hp = read_config("hyperparameter.yaml")

    input_dims = (91, 109, 91)

    # we want one stratified shuffled split
    sss = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=2020)

    job_type_name = "motor-shuffled-500epochs"  # "CV-motor-shuffled" if shuffle_labels else "CV-motor"
    run_name_prefix = "motor-classifier-shuffled" if shuffle_labels else "motor-classifier"

    for fold, (idx_train, idx_valid) in enumerate(sss.split(data_train_full.data, data_train_full.labels)):
        reset_wandb_env()
        wandb_kwargs = {
            "entity": "philis893",
            "project": "thesis",
            "group": "first-steps-motor",
            "name": f"fold-{fold:02d}",
            "job_type": job_type_name if num_folds > 1 else "train",
        }

        save_name = os.path.join("models", job_type_name, wandb_kwargs["name"])
        if os.path.exists(save_name):
            continue

        data_train = torch.utils.data.Subset(data_train_full, idx_train)
        data_valid = torch.utils.data.Subset(data_train_full, idx_valid)

        g = set_random_seed(2020 + fold)

        with wandb.init(config=hp, **wandb_kwargs) as run:

            model_cfg = {
                "channels": [1, 8, 16, 32, 64],
                "lin_neurons": [128, 64],
                "pooling_kernel": 2,
                "kernel_size": run.config.kernel_size,
                "dropout": run.config.dropout,
            }
            model = BrainStateClassifier3d(input_dims, len(class_labels), model_cfg)
            model.to(DEVICE);

            dl_train = DataLoader(data_train, batch_size=run.config.batch_size, shuffle=True, generator=g)
            dl_valid = DataLoader(data_valid, batch_size=run.config.batch_size, shuffle=True, generator=g)
            dl_test = DataLoader(data_test, batch_size=run.config.batch_size, shuffle=False, generator=g)

            best_loss, best_acc = 100, 0
            loss_acc = []
            train_stats, valid_stats = [], []

            # loop for the above set number of epochs
            for epoch in range(run.config.epochs):
                _, _ = model.fit(dl_train, lr=run.config.learning_rate, device=DEVICE)

                # for validating or testing set the network into evaluation mode such that layers like dropout are
                # not active
                with torch.no_grad():
                    tloss, tstats = model.fit(dl_train, device=DEVICE, train=False)
                    vloss, vstats = model.fit(dl_valid, device=DEVICE, train=False)

                # the model.fit() method has 2 output parameters: loss, stats = model.fit()
                # the first parameter is simply the loss for each sample
                # the second parameter is a matrix of n_classes+2-by-n_samples
                # the first n_classes columns are the output probabilities of the model per class
                # the second to last column (i.e., [:, -2]) represents the real labels
                # the last column (i.e., [:, -1]) represents the predicted labels
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
                    "valid_acc": vacc, "valid_loss": vloss,
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

            with torch.no_grad():
                testloss, teststats = model.fit(dl_test, train=False)

            df_test = pd.DataFrame(teststats.tolist(), columns=[*class_labels, *["real", "predicted"]])
            df_test.to_csv(os.path.join(save_name, "test_stats.csv"), index=False)

            testacc = compute_accuracy(teststats[:, -2], teststats[:, -1])
            wandb.run.summary["test_accuracy"] = testacc

            wandb.log({"test_accuracy": testacc, "test_loss": testloss})
            wandb_plots(teststats[:, -2], teststats[:, -1], teststats[:, :-2], class_labels, "test")

            wandb.finish()


if __name__ == '__main__':
    # main(num_folds=10, shuffle_labels=False)
    main(num_folds=10, shuffle_labels=True)
