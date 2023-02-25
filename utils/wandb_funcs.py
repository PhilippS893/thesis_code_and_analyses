import os

import pandas as pd
import wandb


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def get_wandb_csv(entity: str,
                  project: str,
                  group_name: str,
                  keys: list,
                  save_loc: str = ".",
                  overwrite: bool = False) -> pd.DataFrame:
    save_name = os.path.join(save_loc, f"{group_name}.csv")
    if os.path.isfile(save_name) and not overwrite:
        print("File already exist. Loading instead")
        runs_df = pd.read_csv(save_name)
    else:
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs(f"{entity}/{project}", filters={"group_name": {"$regex": f"{group_name}.*"}})

        runs_dict = {}
        """
        important_keys = ['group', 'job_type', 'run_name', 'test_accuracy', 'train_acc', 'valid_acc',
                          'valid_loss', 'best_valid_epoch', 'best_valid_accuracy',
                          'test_loss', 'train_loss']
        """

        for key in keys:
            runs_dict[key] = []

        for run in runs:
            for _, key in enumerate(keys):
                runs_dict[key].extend([run.summary._json_dict[key]]) if key in run.summary._json_dict.keys() else 0

            runs_dict["group"].extend([run.group])
            runs_dict["job_type"].extend([run.job_type])
            runs_dict["run_name"].extend([run.name])

        runs_df = pd.DataFrame(runs_dict)
        runs_df.to_csv(os.path.join(save_loc, f"{group_name}.csv"), index=False)

    return runs_df


def wandb_plots(y_true, y_pred, y_prob, class_labels, dataset):
    wandb.log({
        f"{dataset}-ROC": wandb.plot.roc_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                               title=f"{dataset}-ROC"),
        f"{dataset}-PR": wandb.plot.pr_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                             title=f"{dataset}-PR"),
        f"{dataset}-ConfMat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=class_labels,
                                                          title=f"{dataset}-ConfMat")
    })
