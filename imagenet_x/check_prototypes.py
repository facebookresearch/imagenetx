"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from imagenet_x.utils import (
    load_annotations,
    augment_model_predictions,
    load_model_predictions,
    METACLASSES,
    FACTORS,
)

from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform


def check_prototypes(args):
    model_predictions, top_1_accs = load_model_predictions(args.model_dir)
    annotations = load_annotations(filter_prototypes=False)
    path_to_prototypes = os.path.join(
        args.imagenet_x_dir, f"prototypical_paths.csv"
    )
    prototypes = pd.read_csv(path_to_prototypes)
    annotations_prototypes = annotations[annotations["file_name"].isin(prototypes.file_name)]
    model_annotations= augment_model_predictions(
        annotations_prototypes, model_predictions, args.imagenet_x_dir,
    )

    model_annotations = model_annotations[model_annotations.model!='human']

    accuracies_on_prototypes = model_annotations.groupby('model')['is_correct'].mean().loc[top_1_accs>top_1_accs.loc["resnet50"]]
    print(f"For {len(accuracies_on_prototypes)} models with better accuracies than resnet50, the mean accuracy on the prototypical examples is {accuracies_on_prototypes.mean():.2%} with {accuracies_on_prototypes.std():.5%} standard deviation.")
    print("Full description :")
    print(accuracies_on_prototypes.describe())


def add_check_prototypes_args(parser):
    parser.add_argument("--imagenet-x-dir", default="imagenet_x")
    parser.add_argument(
        "--model-dir",
        nargs="+",
        default="/checkpoint/byoubi/imagenet-testbed",
            # "/checkpoint/byoubi/modelvshuman_predictions",
            # "/checkpoint/byoubi/imagenet-testbed",
            # "/checkpoint/rbalestriero/predictions_csv",
            # "/checkpoint/rbalestriero/color_predictions_csv",
            # "/checkpoint/rbalestriero/color_crop_predictions_csv",
            # "/checkpoint/rbalestriero/adaptive_colorjitter_predictions_csv",
            # "/checkpoint/rbalestriero/smoothing_predictions_csv"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_check_prototypes_args(parser)
    args = parser.parse_args()

    check_prototypes(args)
