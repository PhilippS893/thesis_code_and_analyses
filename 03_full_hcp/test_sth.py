from delphi.networks.ConvNets import BrainStateClassifier3d
from captum.attr import GuidedBackprop
from zennit.rules import Epsilon, Gamma, Pass
from zennit.types import Convolution, Linear, Activation
from zennit.composites import LayerMapComposite
from delphi.utils.tools import save_in_mni
from utils.tools import attribute_with_method
import os
from delphi.utils.tools import z_transform_volume, ToTensor
from torch.utils.data import DataLoader
from delphi.utils.datasets import NiftiDataset
import torch
import numpy as np

composite_lrp_map = [
    (Activation, Pass()),
    (Convolution, Gamma(gamma=.25)),
    (Linear, Epsilon(epsilon=0)),
]

LRP = LayerMapComposite(
    layer_map=composite_lrp_map,
)
LRP.__name__ = 'LRP'


def test_func():
    class_labels = ["footleft", "footright", "handleft", "handright", "tongue",
                    "loss", "reward",
                    "mental", "random",
                    "body", "face", "place", "tool",
                    "match", "relation",
                    "emotion", "neut",
                    "story", "math"]
    label_order = "shuffled"

    fold = 1

    attributor_method = [LRP, GuidedBackprop]

    for i, method in enumerate(attributor_method):

        method_name = str(method.__name__).lower()

        for fold in range(7):
            # load the trained network
            #model = BrainStateClassifier3d(f"models/CV-7folds-{label_order}/seed-2020_fold-{fold:02d}")
            model = BrainStateClassifier3d(f"models/CV-7folds-{label_order}-500epochs/seed-2020_fold-{fold:02d}")
            model.to(torch.device("cpu"));
            model.eval()

            out_dir_name = f"{method_name}/{label_order}/fold-{fold:02d}"
            if not os.path.exists(out_dir_name):
                os.makedirs(out_dir_name)

            for j in range(len(class_labels)):

                print(f"Running {method_name} on {class_labels[j]}")

                out_fname = os.path.join(out_dir_name, '%s.nii.gz' % class_labels[j])

                dl = DataLoader(
                    NiftiDataset('../t-maps/test', [class_labels[j]], 0, device=torch.device("cpu"),
                                 transform=ToTensor()),
                    batch_size=2, shuffle=False, num_workers=0
                )

                for i, (volume, target) in enumerate(dl):
                    attribution = attribute_with_method(method, model, volume, target)

                    subject_attr = np.moveaxis(attribution.squeeze().detach().numpy(), 0, -1)
                    subject_attr = z_transform_volume(subject_attr)
                    avg_attr = subject_attr.mean(axis=-1)

                save_in_mni(subject_attr, out_fname)

                avg_out_name = os.path.join(out_dir_name, "avg")
                if not os.path.exists(avg_out_name):
                    os.makedirs(avg_out_name)
                save_in_mni(avg_attr, os.path.join(avg_out_name, '%s.nii.gz' % class_labels[j]))


if __name__ == '__main__':
    test_func()
