import os
import torch
from glob import glob

import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.masking import apply_mask
from sklearn.feature_selection import mutual_info_regression
from tqdm.notebook import tqdm


def concat_stat_files(folds, file) -> pd.DataFrame:
    '''
    '''
    df_list = []
    for f, fold in enumerate(folds):
        df_list.append(pd.read_csv(os.path.join(fold, file)))

    combined = pd.concat(df_list)

    return combined


def get_test_stats(modelclass, path_to_model, data, class_labels, overwrite=False) -> pd.DataFrame:
    '''
    '''

    stats_file = os.path.join(path_to_model, "test_stats.csv")
    if os.path.isfile(stats_file) and not overwrite:
        stats_df = pd.read_csv(stats_file)

    else:
        model = modelclass(path_to_model)

        with torch.no_grad():
            _, stats = model.fit(data, train=False)

        stats_df = pd.DataFrame(stats.tolist(), columns=[*class_labels, *["real", "predicted"]])
        stats_df.to_csv(os.path.join(path_to_model, "test_stats.csv"), index=False)

    return stats_df


def attribute_with_method(attributor_method, model, data, target):
    data.requires_grad = True

    attributor_name = str(attributor_method.__name__)

    if attributor_name == 'LRP':
        grad_dummy = torch.eye(model.config['n_classes'], device=torch.device("cpu"))[target]

        with attributor_method.context(model) as modified_model:
            output = modified_model(data)
            attribution, = torch.autograd.grad(output, data, grad_outputs=grad_dummy)

    elif attributor_name == 'GuidedBackprop':
        gbp = attributor_method(model)
        attribution = gbp.attribute(data, target)

    return attribution


def compute_mi(
        task_label: str,
        class_labels: list,
        mask: str,
        save_prefix: str,
        n_folds: int = 10,
        attr_methods: list = ["lrp", "guidedbackprop"],
        contrast: str = "grpattr-vs-grporig",
        seed: int = 2020,
) -> pd.DataFrame:
    """

    :param task_label:
    :param class_labels:
    :param mask:
    :param save_prefix:
    :param n_folds:
    :param attr_methods:
    :param contrast:
    :return:
    """

    allowed_contrasts = ["subattr-vs-subt", "attr-real-vs-shuffled",
                         "subattr-vs-grporig", "grpattr-vs-grporig",
                         "subattr-vs-grpattr", "subt-vs-grpt"]
    assert contrast in allowed_contrasts, \
        f'value in contrast: {contrast} does not match one of {allowed_contrasts}'

    mis_list = []
    msk = load_img(mask)
    out_name = os.path.join("stats", f"{save_prefix}_{contrast}.csv")

    if os.path.isfile(out_name):
        print("{} already exists. Loading instead.".format(out_name))
        mis = pd.read_csv(out_name)
    else:
        if contrast == allowed_contrasts[5]:

            # case subt-vs-grpt
            for _, lbl in enumerate(class_labels):
                X = apply_mask(sorted(glob(os.path.join("../t-maps", "test", lbl, "*sub*.nii.gz"))), msk)

                grp_maps = sorted(glob(os.path.join("stat-maps", "orig", "left-out", f"{lbl}*wo*z_score.nii.gz")))
                mi = np.zeros(len(X))
                for i, grp_wo_sub in tqdm(enumerate(grp_maps)):
                    y = apply_mask(grp_wo_sub, msk)
                    mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y,
                                                   discrete_features=False, random_state=seed)
                mis_list.append(pd.DataFrame({"mi": mi.tolist(), "class": lbl, "attr_method": "tscore",
                                              "contrast": contrast}))

        else:
            for _, lbl in enumerate(class_labels):

                if "grporig" in contrast:
                    print(f"Loading stat-maps/orig/{lbl}_z_score.nii.gz")
                    y = apply_mask(f"stat-maps/orig/{lbl}_z_score.nii.gz", msk)

                for m, method in enumerate(attr_methods):

                    for fold in tqdm(range(n_folds)):

                        if contrast == allowed_contrasts[0]:

                            # 1. load the data of the respective fold
                            print(
                                f"Loading {method}/real/{task_label}_fold-{fold:02d}/{lbl}.nii.gz and" \
                                f"../t-maps/test/{lbl}/sub*.nii.gz")
                            X = apply_mask(f"{method}/real/{task_label}_fold-{fold:02d}/{lbl}.nii.gz", msk)
                            y = apply_mask(glob(f"../t-maps/test/{lbl}/sub*.nii.gz"), msk)

                            # 2. compute the mi for each subject and save the score
                            mi = np.zeros(len(X))

                            for i in range(len(X)):
                                mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y[i, :],
                                                               discrete_features=False, random_state=seed)

                        elif contrast == allowed_contrasts[1]:

                            # 1. load the data of the respective fold
                            print(
                                f"Loading {method}/shuffled/{task_label}_fold-{fold:02d}/{lbl}.nii.gz" \
                                f"and {method}/real/{task_label}_fold-{fold:02d}/{lbl}.nii.gz"
                            )
                            X = apply_mask(f"{method}/shuffled/{task_label}_fold-{fold:02d}/{lbl}.nii.gz", msk)
                            y = apply_mask(f"{method}/real/{task_label}_fold-{fold:02d}/{lbl}.nii.gz", msk)

                            # 2. compute the mi for each subject and save the score
                            mi = np.zeros(len(X))

                            for i in range(len(X)):
                                mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y[i, :],
                                                               discrete_features=False, random_state=seed)

                        elif contrast == allowed_contrasts[2]:

                            print(f"Loading {method}/real/{task_label}_fold-{fold:02d}/{lbl}.nii.gz")
                            X = apply_mask(f"{method}/real/{task_label}_fold-{fold:02d}/{lbl}.nii.gz", msk)
                            mi = mutual_info_regression(X=X.reshape(-1, 1), y=y,
                                                        discrete_features=False, random_state=seed)

                        elif contrast == allowed_contrasts[3]:

                            print(f"Loading stat-maps/{method}/real/{task_label}_fold-{fold:02d}/{lbl}_z_score.nii.gz")
                            X = apply_mask(
                                os.path.join(f"stat-maps/{method}/real/{task_label}_fold-{fold:02d}/{lbl}_z_score.nii.gz"),
                                msk
                            )
                            mi = mutual_info_regression(X=X.reshape(-1, 1), y=y,
                                                        discrete_features=False, random_state=seed)

                        elif contrast == allowed_contrasts[4]:
                            # case subattr-vs-grpattr
                            X = apply_mask(
                                os.path.join(method, "real", f"{task_label}_fold-{fold:02d}", f"{lbl}.nii.gz"),
                                msk
                            )

                            grp_maps = sorted(glob(os.path.join("stat-maps", method, "real",
                                                         f"{task_label}_fold-{fold:02d}", "left-out",
                                                         f"{lbl}*wo*z_score.nii.gz")))
                            mi = np.zeros(len(X))
                            for i, grp_wo_sub in enumerate(grp_maps):
                                y = apply_mask(grp_wo_sub, msk)
                                mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y,
                                                               discrete_features=False, random_state=seed)

                        mis_list.append(
                            pd.DataFrame({"mi": mi.tolist(), "class": lbl, "fold": fold,
                                          "attr_method": method, "contrast": contrast}))

        mis = pd.concat(mis_list)
        mis.to_csv(out_name, index=False)

    return mis
