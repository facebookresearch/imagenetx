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

from imagenet_x.utils import (
    load_annotations,
    augment_model_predictions,
    load_model_predictions,
    METACLASSES,
    FACTORS,
)

from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

def escape(l):
    return [x.replace("_", " ") for x in l]


def estimated_calibration_error(corrects, probabilities, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], probabilities)

    bin_sums = np.bincount(binids, weights=probabilities, minlength=len(bins))
    bin_correct = np.bincount(binids, weights=corrects, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    avg_correct_in_bin = bin_correct[nonzero] / bin_total[nonzero]
    avg_confidence_in_bin = bin_sums[nonzero] / bin_total[nonzero]
    prob_in_bin = bin_total[nonzero] / len(corrects)

    ece = np.sum(np.abs(avg_correct_in_bin - avg_confidence_in_bin) * prob_in_bin)

    return ece


def _compute_calibration(df, error_type="is_correct", groupby="model"):
    return df.groupby(groupby).apply(
        lambda x: estimated_calibration_error(x[error_type], x["predicted_probability"])
    )


def _compute_factor_calibration_errors(all_df, factors, error_type, groupby="model"):
    calibration = pd.DataFrame()
    for factor in factors:
        calibration[factor] = _compute_calibration(
            all_df[all_df[factor] == 1], error_type=error_type, groupby=groupby
        )
    calibration["worst_factor"] = calibration.max(axis=1)
    calibration["average"] = _compute_calibration(
        all_df, error_type=error_type, groupby=groupby
    )
    return calibration


def compute_factor_calibration_errors(all_df, factors, error_type):
    return _compute_factor_calibration_errors(
        all_df, factors, error_type, groupby="model"
    )


def compute_factor_metaclass_calibration_errors(
    all_df, factors, error_type, metaclasses
):
    return _compute_factor_calibration_errors(
        all_df, factors, error_type, groupby=["model", "metaclass"]
    ).loc[:, metaclasses, :]


def _compute_factor_accuracies(all_df, factors, error_type, groupby="model"):
    # Since factors are represented in a one hot manner, we can compute the count of correct predictions for each factor
    # by grouping by model and summing over the 0 axis
    factor_correct = all_df[all_df[error_type]].groupby(groupby)[factors].sum()
    groups = all_df.groupby(groupby)
    factor_total = groups[factors].sum()
    factor_accs = factor_correct / factor_total
    factor_accs["worst_factor"] = factor_accs.min(axis=1)
    factor_accs["average"] = groups[error_type].mean()

    return factor_accs


def compute_factor_accuracies(
    all_df,
    factors,
    error_type = "is_correct",
):
    return _compute_factor_accuracies(all_df, factors, error_type, groupby="model")


def compute_factor_metaclass_accuracies(all_df, factors, error_type, metaclasses):
    return _compute_factor_accuracies(
        all_df, factors, error_type, groupby=["model", "metaclass"]
    ).loc[:, metaclasses, :]

def worst_k_class_acc(all_df, k=50, error_type="is_correct"):
    worst_classes = (
        all_df.groupby(["model", "class"])[error_type]
        .mean()
        .sort_values()
        .groupby(level="model")
        .head(k)
        .index
    )
    worst_classes_accs = (
        all_df.set_index(["model", "class"])
        .loc[worst_classes]
        .groupby("model")[error_type]
        .mean()
    )
    return worst_classes_accs


def spearman_corr_heatmaps(annotations, factors, metaclasses):
    if metaclasses:
        superclass_onehot = pd.get_dummies(annotations["metaclass"])
        annotations = pd.concat([annotations, superclass_onehot], axis=1)
        all_cols = factors + metaclasses
    else:
        all_cols = factors
    annotations = annotations[all_cols]

    m = annotations.values.astype(int).T
    corr = squareform(pdist(m, metric=lambda x, y: spearmanr(x, y)[0]))
    p_val = squareform(pdist(m, metric=lambda x, y: spearmanr(x, y)[1]))

    return pd.DataFrame(corr, index=all_cols, columns=all_cols), pd.DataFrame(
        p_val, index=all_cols, columns=all_cols
    )


def get_factor_accuracies(
    model_dir,
    which_factor = "top",
    partition = "val",
    filter_prototypes=True,
    error_type = "class",
):
    """Utility function that computes the accuracy of each factor for a list of models.

    Args:
        model_dir (str): Path to the model predictions directory.
        which_factor (str, optional): Which factor to use. Defaults to "top".
        partition (str, optional): Which partition to use. Defaults to "val".
        filter_prototypes (bool, optional): Whether to filter out prototypes. Defaults to True.
        error_type (str, optional): Which error type to use, available options are ["class", "metaclass", "real_class"]. Defaults to "class".

    Returns:
        pd.DataFrame: Dataframe with the accuracy of each factor as columns and models as indices.
    """
    
    error_type = {
        "real_class": "is_correct_real",
        "metaclass": "is_metaclass_correct",
        "class": "is_correct",
    }[error_type]
    annotations = load_annotations(
        which_factor=which_factor,
        partition=partition,
        filter_prototypes=filter_prototypes,
    )
    model_predictions = load_model_predictions(model_dir)
    augmented_predictions = augment_model_predictions(annotations, model_predictions[0])
    return compute_factor_accuracies(
        augmented_predictions, FACTORS, error_type=error_type
    )

def error_ratio(result_df):
    return (1-result_df[FACTORS]).divide((1-result_df["average"]), axis=0)

def generate_all_tables(args):
    for model_dir in args.model_dirs:
        model_predictions, top_1_accs = load_model_predictions(model_dir, verbose=args.verbose)
        for partition in args.partitions:
            for which_factor in args.which_factor:
                annotations = load_annotations(
                    which_factor=which_factor,
                    partition=partition,
                    filter_prototypes=not args.keep_prototypes,
                )
                if partition == "val":
                    model_annotations_all = augment_model_predictions(
                        annotations,
                        model_predictions,
                        add_human_performance=True,
                    )

                    for error_type in args.error_types:
                        if error_type != "real_class":
                            model_annotations = model_annotations_all[model_annotations_all.model!='human']
                        else:
                            model_annotations = model_annotations_all
                        table_dir = (
                            Path(args.table_dir)
                            / partition
                            / f"{which_factor}_factor"
                            / model_dir.split("/")[-1]
                            / error_type
                        )
                        os.makedirs(table_dir, exist_ok=True)

                        error_type = {
                            "real_class": "is_correct_real",
                            "metaclass": "is_metaclass_correct",
                            "class": "is_correct",
                        }[error_type]

                        k_classes = 100
                        worst_classes_accs = worst_k_class_acc(
                            model_annotations, k=k_classes, error_type=error_type
                        )

                        print(
                            "Generating accuracy table for",
                            error_type,
                            "with",
                            which_factor,
                            "factors",
                        )

                        x = compute_factor_accuracies(
                            model_annotations, FACTORS, error_type
                        )

                        x[f"worst_{k_classes}_classes"] = worst_classes_accs
                        x.to_csv(table_dir / f"accuracies_per_factor.csv")

                        x = compute_factor_metaclass_accuracies(
                            model_annotations, FACTORS, error_type, METACLASSES
                        )
                        x.to_csv(table_dir / f"accuracies_per_factor_metaclass.csv")


def add_aggregate_args(parser):
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=[
            "model_predictions/base",
            "model_predictions/imagenet_testbed",
            "model_predictions/modelvshuman",
            "model_predictions/crop",
            "model_predictions/colorjitter",
            "model_predictions/gaussian_blur",
            "model_predictions/colorjitter_crop",
            "model_predictions/adaptive_colorjitter_crop",
        ],
    )
    parser.add_argument("--table-dir", default="results")
    parser.add_argument(
        "--error-types",
        nargs="+",
        default=["class", "metaclass", "real_class"],
        choices=["class", "metaclass", "real_class"],
    )
    parser.add_argument(
        "--which-factor", nargs="+", default=["top", "multi"], choices=["multi", "top"]
    )
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "val"], choices=["train", "val"]
    )
    parser.add_argument("--keep-prototypes", action="store_true")
    parser.add_argument("--verbose", action="store_true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_aggregate_args(parser)
    args = parser.parse_args()

    generate_all_tables(args)
