"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
from tqdm import tqdm

import imagenet_x.annotations

METACLASSES = [
    "device",
    "dog",
    "commodity",
    "bird",
    "structure",
    "covering",
    "wheeled_vehicle",
    "food",
    "equipment",
    "insect",
    "vehicle",
    "furniture",
    "primate",
    "vessel",
    "snake",
    "natural_object",
    "other",
]

FACTORS = [
    "pose",
    "background",
    "pattern",
    "color",
    "smaller",
    "shape",
    "partial_view",
    "subcategory",
    "texture",
    "larger",
    "darker",
    "object_blocking",
    "person_blocking",
    "style",
    "brighter",
    "multiple_objects",
]


def is_correct_real(file_names, predicted_class, path_to_real_labels):
    real_labels_json = json.load(open(path_to_real_labels))
    real_labels = {
        f"ILSVRC2012_val_{(i+1):08d}.JPEG": labels
        for i, labels in enumerate(real_labels_json)
    }
    real_labels = pd.DataFrame.from_dict(real_labels, orient="index")
    real_labels = real_labels.reindex(file_names).reset_index(drop=True)

    return real_labels.eq(predicted_class, axis=0).any(axis=1)


def to_metaclass(series, path_to_metaclass_mapping):
    metaclass_mapping = pd.read_csv(path_to_metaclass_mapping)
    mask = metaclass_mapping.isin(METACLASSES)
    metaclass_mapping = metaclass_mapping.where(mask, "other")
    return metaclass_mapping.name.loc[series].reset_index(drop=True)


def label_ids_to_name(series, path_to_imagenet_labels):
    labels = pd.read_csv(path_to_imagenet_labels, names=["wnid", "label"])
    return labels.loc[series].label.set_axis(series.index)


def sanity_check(annotations, all):
    assert (
        annotations.file_name.is_unique
    ), "The annotations dataset has multiple annotations for some images"
    for model in all:
        assert all[
            model
        ].file_name.is_unique, (
            f"The model {model} predictions has multiple predictions for some images"
        )


def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

def get_annotation_path():
    return files("imagenet_x") / "annotations" 

def load_model_predictions(models_dir: str, verbose=False):
    filename_label = pd.read_csv(get_annotation_path() / "filename_label.csv")
    paths, labels = (
        filename_label.file_name,
        filename_label.set_index("file_name").label,
    )

    models = {}
    top_1_accs = pd.Series(dtype=np.float32)
    model_dirs = os.listdir(models_dir)
    assert len(model_dirs) > 0, "No models found in models_dir"
    for path in tqdm(model_dirs, desc="Loading model predictions", disable=not verbose):
        df = pd.DataFrame()
        df["file_name"] = paths
        # If model predictions is stored in a pickle (such as Imagenet testbed predictions)
        if path.endswith(".pickle"):
            with open(os.path.join(models_dir, path), "rb") as f:
                preds = pickle.load(f)
                model = preds["model"]
                logits = preds["logits"]
                df["predicted_class"] = logits.argmax(axis=1)
                df["predicted_probability"] = softmax(logits, axis=1).max(axis=1)
                df = df.set_index("file_name")
        elif path.endswith(".csv"):
            df = pd.read_csv(os.path.join(models_dir, path)).set_index("file_name")
            model = Path(path).stem
        elif path.endswith("DS_Store"):
            continue
        else:
            raise ValueError(f"Unknown file type {path}")

        top_1_accs[model] = (df["predicted_class"] == labels).mean()
        models[model] = df
    return models, top_1_accs


def load_annotations(
    which_factor="top",
    partition="val",
    filter_prototypes=True,
):
    """Loads the annotations for the given partition and factor selection.

    Parameters
    ----------
    which_factor : str, optional
        Which factors to use for the annotations, either "top" or "multi", by default "top"
    partition : str, optional
        Which partition to use, either "train" or "val", by default "val"
    filter_prototypes : bool, optional
        Whether to filter out the prototype images, by default True

    Returns
    -------
    pd.DataFrame
        The annotations for the given partition and factor selection
    """
    imagenet_x_type = f"imagenet_x_{partition}_{which_factor}_factor.jsonl"
    annot_file = get_annotation_path()
    imagenet_x_json = annot_file / imagenet_x_type
    metaclass_mapping = annot_file / f"imagenet_1k_classes_to_100_metaclasses.csv"
    prototypes = annot_file / f"prototypical_paths.csv"
    annotations = pd.read_json(imagenet_x_json, lines=True)
    annotations["metaclass"] = to_metaclass(annotations["class"], metaclass_mapping)
    if filter_prototypes:
        prototypes = pd.read_csv(prototypes)
        annotations = annotations[~annotations["file_name"].isin(prototypes.file_name)]
    return annotations


def augment_model_predictions(
    annotations, models, add_human_performance=False, verbose=False
):
    path_to_real_labels = get_annotation_path() / "imagenet_real_labels.jsonl"
    path_to_metaclass_mapping = (
        get_annotation_path() / "imagenet_1k_classes_to_100_metaclasses.csv"
    )

    if add_human_performance:
        human_model = annotations[["file_name", "class"]].rename(
            columns={"class": "predicted_class"}
        )
        human_model["predicted_probability"] = 1.0

        models["human"] = human_model

    all = {
        model: pd.merge(annotations, models[model], on="file_name", how="inner")
        for model in tqdm(
            models,
            desc="Merging model predictions with annotations",
            disable=not verbose,
        )
    }

    for model in all:
        all[model]["model"] = model

    if verbose:
        print("Sanity check")
    sanity_check(annotations, all)

    if verbose:
        print("Concatenation")
    model_annotations = pd.concat(all.values(), ignore_index=True)

    if verbose:
        print("Adding is correct column")
    model_annotations["is_correct"] = (
        model_annotations["predicted_class"] == model_annotations["class"]
    )

    if verbose:
        print("Adding is real correct column")
    model_annotations["is_correct_real"] = is_correct_real(
        model_annotations.file_name,
        model_annotations["predicted_class"],
        path_to_real_labels,
    )

    if verbose:
        print("Adding predicted metaclass")
    model_annotations["predicted_metaclass"] = to_metaclass(
        model_annotations["predicted_class"], path_to_metaclass_mapping
    )

    if verbose:
        print("Adding is correct metaclass column")
    model_annotations["is_metaclass_correct"] = (
        model_annotations["predicted_metaclass"] == model_annotations["metaclass"]
    )

    return model_annotations
