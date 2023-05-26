import os
import torch
from glob import glob

import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.masking import apply_mask
from sklearn.feature_selection import mutual_info_regression
from tqdm.notebook import tqdm
from delphi import mni_template


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
            attribution, = torch.autograd.grad(output.squeeze(), data, grad_outputs=grad_dummy)

    elif attributor_name == 'GuidedBackprop':
        gbp = attributor_method(model)
        attribution = gbp.attribute(data, target)

    return attribution


def compute_mi(
        class_labels: list,
        mask: str,
        save_prefix: str,
        fold: int = 0,
        attr_methods: list = ["lrp", "guidedbackprop"],
        contrast: str = "grpattr-vs-grporig",
        seed: int = 2020,
) -> pd.DataFrame:
    """

    :param class_labels:
    :param mask:
    :param save_prefix:
    :param fold:
    :param attr_methods:
    :param contrast:
    :param seed:
    :return:
    """

    allowed_contrasts = ["subattr-vs-subt", "attr-real-vs-shuffled",
                         "subattr-vs-grpt", "grpattr-vs-grpt",
                         "subattr-vs-grpattr", "subt-vs-grpt",
                         "model-trained-vs-rnd"]
    assert contrast in allowed_contrasts, \
        f'value in contrast: {contrast} does not match one of {allowed_contrasts}'

    mis_list = []
    msk = load_img(mask)
    out_name = os.path.join("stats", f"{save_prefix}_{contrast}.csv")

    if os.path.isfile(out_name):
        print("{} already exists. Loading instead.".format(out_name))
        mis = pd.read_csv(out_name)
    else:
        print(f"Computing MI for {contrast}")
        if contrast == allowed_contrasts[5]:

            # case subt-vs-grpt
            for _, lbl in tqdm(enumerate(class_labels), desc="label"):
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
            for _, lbl in tqdm(enumerate(class_labels), desc="label"):

                if "grpt" in contrast:
                    y = apply_mask(f"stat-maps/orig/{lbl}_z_score.nii.gz", msk)

                for m, method in enumerate(attr_methods):

                    if contrast == allowed_contrasts[0]:

                        # 1. load the data of the respective fold
                        X = apply_mask(f"{method}/real/fold-{fold:02d}/{lbl}.nii.gz", msk)
                        y = apply_mask(glob(f"../t-maps/test/{lbl}/sub*.nii.gz"), msk)

                        # 2. compute the mi for each subject and save the score
                        mi = np.zeros(len(X))

                        for i in range(len(X)):
                            mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y[i, :],
                                                           discrete_features=False, random_state=seed)

                    elif contrast == allowed_contrasts[1]:

                        # 1. load the data of the respective fold
                        X = apply_mask(f"{method}/shuffled/fold-{fold:02d}/{lbl}.nii.gz", msk)
                        y = apply_mask(f"{method}/real/fold-{fold:02d}/{lbl}.nii.gz", msk)

                        # 2. compute the mi for each subject and save the score
                        mi = np.zeros(len(X))

                        for i in range(len(X)):
                            mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y[i, :],
                                                           discrete_features=False, random_state=seed)

                    elif contrast == allowed_contrasts[2]:
                        # subattr-vs-grpt

                        X = apply_mask(f"{method}/real/fold-{fold:02d}/{lbl}.nii.gz", msk)
                        mi = np.zeros(len(X))
                        for i in range(len(X)):
                            mi[i] = mutual_info_regression(X=X[i, :].reshape(-1, 1), y=y,
                                                        discrete_features=False, random_state=seed)

                    elif contrast == allowed_contrasts[3]:

                        X = apply_mask(
                            os.path.join(f"stat-maps/{method}/real/fold-{fold:02d}/{lbl}_z_score.nii.gz"),
                            msk
                        )
                        mi = mutual_info_regression(X=X.reshape(-1, 1), y=y,
                                                    discrete_features=False, random_state=seed)

                    elif contrast == allowed_contrasts[4]:
                        # case subattr-vs-grpattr
                        X = apply_mask(
                            os.path.join(method, "real", f"fold-{fold:02d}", f"{lbl}.nii.gz"),
                            msk
                        )

                        grp_maps = sorted(glob(os.path.join("stat-maps", method, "real",
                                                            f"fold-{fold:02d}", "left-out",
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


def load_data(n, data_dir, classes, split, brain_mask=None):
    for i, label in tqdm(enumerate(classes), desc="class"):
        file_list = glob.glob(os.path.join(data_dir, split, label, 'sub*.nii.gz'))
        file_list = file_list[:n]
        if i == 0:
            data = apply_mask(load_img(file_list), brain_mask)
        else:
            data = np.vstack((data, apply_mask(load_img(file_list), brain_mask)))

    return data


def run_pca(data, max_components, batch_size, save_dir="./pca_models", prefix=None, seed=2020):
    """
    IncrementalPCA is helpful for datasets that require large amounts of memory.
    max_components: the number of components you want to keep
    batch_size: how many samples per batch
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, '%s%dsamples_seed%d.pkl' % (prefix, max_components, seed))

    if os.path.exists(save_name):
        print('%s already exists. Loading instead' % save_name)
        with open(save_name, 'rb') as pickle_file:
            pca = pk.load(pickle_file)
    else:
        pca = IncrementalPCA(n_components=max_components, batch_size=batch_size)
        pca.fit(data)
        print('Saving %s for later use.' % save_name)
        with open(save_name, 'wb') as pickle_file:
            pk.dump(pca, pickle_file)

    return pca


def run_svm(data, Y, approach="ovr", n_classes=None, prefix=None, dims=None, save_dir="./svm_models", seed=2020):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if dims is None:
        dims = data.shape[1]

    save_name = os.path.join(save_dir, "%s%s_%dsubs_%ddims_seed%d.pkl" % (
        prefix, approach, data.shape[0] / n_classes, dims, seed))

    if os.path.exists(save_name):
        print('%s already exists. Loading instead' % save_name)
        with open(save_name, 'rb') as pickle_file:
            clf = pk.load(pickle_file)
    else:
        # ONE-VS-ONE
        if approach == "ovo":
            clf = OneVsOneClassifier(
                SVC(kernel="linear", probability=True, break_ties=True, random_state=seed)).fit(data[:, :dims], Y)

        # ONE-VS-REST
        elif approach == "ovr":
            clf = OneVsRestClassifier(
                SVC(kernel="linear", probability=True, break_ties=True, random_state=seed)).fit(data[:, :dims], Y)

        with open(save_name, 'wb') as pickle_file:
            pk.dump(clf, pickle_file)

    return clf


def save_maps(clf, data, approach, nsubs, dims, prefix=None, invert_pca=False, save_dir="./svm_maps", seed=2020,
              brain_mask=mni_template):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name = os.path.join(save_dir, "%s%s_%dsubs_%ddims_seed%d.nii.gz" % (prefix, approach, nsubs, dims, seed))
    if os.path.exists(save_name):
        pass
    else:
        print("saving %s" % save_name)
        n_estimators = len(clf.estimators_)
        to_save = np.zeros((n_estimators, data.shape[-1]))

        if not os.path.exists(save_name):
            for i in range(n_estimators):
                if not invert_pca:
                    to_save[i] = clf.estimators_[i].coef_
                else:
                    to_save[i] = np.dot(clf.estimators_[i].coef_, pca.components_[:dims])

        mu = to_save.mean(axis=1)
        mu = mu[:, np.newaxis]
        std = to_save.std(axis=1)
        std = std[:, np.newaxis]
        with np.errstate(divide='ignore'):
            z_transformed = (to_save - mu) / std
        img = unmask(z_transformed, brain_mask)
        save(img, save_name)


def predict(clf, data, Y, dims=None):
    if dims is None:
        dims = data.shape[1]

    preds = clf.predict(data[:, :dims])
    accs = (np.sum(preds == Y) / len(Y))

    return preds, accs
