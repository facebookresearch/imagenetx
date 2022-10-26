"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pandas as pd

try:
    from utils import (
        load_model_predictions,
        load_annotations,
        augment_model_predictions,
        FACTORS,
        METACLASSES,
    )
except ImportError:
    from .utils import (
        load_model_predictions,
        load_annotations,
        augment_model_predictions,
        FACTORS,
        METACLASSES,
    )


class Prevalence:
    """Measures the prevalence/shiftsof factors.

    Args:
        imagenet_x_dir: directory containing imagenet_x annotations
        model_dir: directory containing model predictions
        select_top_factor: whether to compute shifts only for the top factor

    Attributes:
        df: dataframe containing top factor for each image
        metaclass_prevalence: dataframe of the prevalence of each factor per metaclass
        metaclass_prevalence_shift: dataframe of the prevalence of each factor per metaclass
            relative to their overall prevalence.
        metaclass_misclassification_shifts_df: dataframe containing shifts by model and metaclass
        misclassification_shifts_df: dataframe containing aggregate shifts by model

    """

    def __init__(
        self,
        imagenet_x_dir: str = "imagenet_x/",
        models_dir: str = "models/",
        which_factor: str = "top",
    ):
        self.models, _ = load_model_predictions(models_dir)
        self.annotations = load_annotations()
        self.all_df = augment_model_predictions(self.annotations, self.models)

        self.factors = FACTORS
        self.metaclasses = METACLASSES
        self.group = "metaclass"

        # model agnostic
        self.metaclass_prevalence = self.compute_metaclass_prevalence()
        self.metaclass_prevalence_shift = self.compute_metaclass_prevalence_shift()

        # dataframe containing shifts by factor for every meta class and model
        self.metaclass_misclassification_shifts_df = (
            self.compute_all_metaclass_misclassification_shifts()
        )

        self.misclassification_shifts_df = self.compute_all_misclassification_shifts()

    def compute_all_metaclass_misclassification_shifts(self) -> pd.DataFrame:
        dfs = []

        for model in self.models:
            model_df = self.compute_metaclass_misclassification_shift(
                model
            ).reset_index()
            model_df["model"] = model
            dfs.append(model_df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def compute_all_misclassification_shifts(self) -> pd.DataFrame:
        dfs = []

        for model in self.models:
            model_df = self.compute_misclassification_shift(model)
            dfs.append(model_df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def compute_metaclass_prevalence(self) -> pd.DataFrame:
        """Index is metaclass and prevalence proportion for each factor in columns.
        Each metaclass row should sum to 1.
        """
        df = self.annotations
        group_counts = df.groupby(self.group)[self.factors].sum()
        prevalence = group_counts.divide(group_counts.sum(axis=1), axis=0)
        return prevalence

    def compute_metaclass_prevalence_shift(self) -> pd.DataFrame:
        """Index is metaclass and prevalence proportion shift for each factor in columns.
        Relative to the aggregate prevalence for each factor
        """
        df = self.annotations
        total_counts = df[self.factors].sum()
        total_prevalence = total_counts / total_counts.sum()

        group_counts = df.groupby(self.group)[self.factors].sum()
        prevalence = group_counts.divide(group_counts.sum(axis=1), axis=0)

        shift = (prevalence - total_prevalence).divide(total_prevalence)
        shift = shift.fillna(0)
        return shift

    def compute_metaclass_misclassification_shift(self, model: str) -> pd.DataFrame:
        df = self.all_df[self.all_df.model == model]
        misclassified_counts = (
            df[df.is_correct == False].groupby(self.group)[self.factors].sum()
        )
        misclassified_prevalence = misclassified_counts.divide(
            misclassified_counts.sum(axis=1), axis=0
        )

        shift = (misclassified_prevalence - self.metaclass_prevalence).divide(
            self.metaclass_prevalence
        )
        shift = shift.fillna(0)
        accuracy = self.compute_metaclass_accuracy(df)
        shift = shift.merge(accuracy, on=self.group)

        return shift

    def compute_metaclass_accuracy(self, model_df: pd.DataFrame) -> pd.Series:
        """Computes the accuracy for data frame"""
        df = model_df
        total = df.groupby(self.group)["file_name"].count()
        correct = df[df["is_correct"] == True].groupby(self.group)["file_name"].count()
        accuracy = correct.divide(total).rename("accuracy")
        return accuracy

    def compute_misclassification_shift(self, model: str) -> pd.DataFrame:
        df = self.all_df[self.all_df.model == model]
        misclassified_counts = df[df.is_correct == False][self.factors].sum()
        misclassified_prevalence = misclassified_counts.divide(
            misclassified_counts.sum()
        )

        total_counts = df[self.factors].sum()
        prevalence = total_counts.divide(total_counts.sum())

        shift = (misclassified_prevalence - prevalence).divide(prevalence)
        shift = shift.fillna(0)

        # turn into dataframe
        df = shift.reset_index()
        df["model"] = model
        df = df.rename(columns={0: "shift", "index": "factor"})
        return df
