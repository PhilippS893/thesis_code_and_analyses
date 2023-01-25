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


def wandb_plots(y_true, y_pred, y_prob, class_labels, dataset):
    wandb.log({
        f"{dataset}-ROC": wandb.plot.roc_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                               title=f"{dataset}-ROC"),
        f"{dataset}-PR": wandb.plot.pr_curve(y_true=y_true, y_probas=y_prob, labels=class_labels,
                                             title=f"{dataset}-PR"),
        f"{dataset}-ConfMat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, class_names=class_labels,
                                                          title=f"{dataset}-ConfMat")
    })
