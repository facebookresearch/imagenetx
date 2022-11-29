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

try:
    import plotly.graph_objects as go
    from matplotlib import pyplot as plt
    import matplotlib.ticker as ticker
    from plotly.subplots import make_subplots
    import plotly.express as px
    import seaborn as sns
    from scipy import stats
except ImportError:
    raise ImportError("Please install plotly and seaborn and scipy to use this module.")


from imagenet_x.utils import load_annotations, FACTORS, METACLASSES
from imagenet_x.prevalence import Prevalence
from imagenet_x.aggregate import spearman_corr_heatmaps
from imagenet_x.model_types import model_types_map, SELFSUPERVISED_MODELS, ModelTypes


def escape(l):
    return [x.replace("_", " ") for x in l]

colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

MODEL_TYPE_ORDER = [
    ModelTypes.MORE_DATA.value,
    ModelTypes.STANDARD.value,
    ModelTypes.SELFSUPERVISED.value,
    ModelTypes.ROBUST_INTV.value,
]


def factor_distribution_comparison(annotations_to_compare, fname=None, remove_underscore=True):
    # sns.set(style="whitegrid")
    plt.figure(figsize=(8, 1.5))

    all_distribs = []
    for (split, factor_selection), df in annotations_to_compare.items():
        counts = df[FACTORS].sum(axis=0)
        distrib = counts / len(df)
        distrib.index.name = "Factor"
        distrib = distrib.to_frame("Percentage").reset_index()
        factor_selection = (
            "all factors" if factor_selection == "multi" else "top factor"
        )
        distrib["annotation"] = split + " " + factor_selection
        all_distribs.append(distrib)
    all_distribs = pd.concat(all_distribs)
    # all_distribs.Factor = all_distribs.Factor.apply(lambda x: x.replace('_', ' '))
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    # sns.despine()

    for i, factors in enumerate(
        [
            all_distribs.groupby("Factor")
            .Percentage.mean()
            .sort_values(ascending=False)
            .index.values[4 * i : 4 * (i + 1)]
            for i in range(4)
        ]
    ):
        all_distribs.loc[all_distribs.Factor.isin(factors), "group"] = i
    g = sns.FacetGrid(
        all_distribs, col="group", sharex=False, sharey=False, height=2.5, aspect=0.9
    )
    if remove_underscore:
        all_distribs.replace(r"_", " ", regex=True, inplace=True)
    g.map_dataframe(
        sns.barplot,
        x="Factor",
        y="Percentage",
        data=all_distribs,
        palette=colors,
        hue="annotation",
        hue_order=[
            "train all factors",
            "val all factors",
            "train top factor",
            "val top factor",
        ],
    )
    g.add_legend()
    sns.move_legend(
        g, "lower center", bbox_to_anchor=(0.5, 1), ncol=4, title=None, frameon=False
    )

    def style(*args, **kwargs):
        sns.despine()
        plt.xticks(rotation=30, ha="right")
        # plt.yscale("log")
        # sns.set_palette(sns.color_palette(colors))
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        plt.gca().set_xlabel("")
        plt.gca().set_axisbelow(True)
        plt.grid(axis="y", which="major", linewidth=1, alpha=0.3, linestyle="--")
        # plt.gca().yaxis.set_minor_formatter(ticker.PercentFormatter(xmax=1))

    g.map_dataframe(style)
    for ax in g.axes.flat:
        ax.set_title("")

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", transparent=True)
    plt.close()


def active_factor_distribution_comparison(annotations_to_compare, fname=None, remove_underscore=True):
    # sns.set(style="whitegrid")
    plt.figure(figsize=(5, 2))

    all_distribs = []
    for split in ["val", "train"]:
        df = annotations_to_compare[(split, "multi")]
        distrib = df[FACTORS].sum(axis=1).value_counts()
        distrib.index.name = "Number of active factors"
        distrib = distrib.to_frame("Count").reset_index()
        distrib["annotation"] = split
        all_distribs.append(distrib)
    all_distribs = pd.concat(all_distribs)
    if remove_underscore:
        all_distribs.replace(r"_", " ", regex=True, inplace=True)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    g = sns.barplot(
        x="Number of active factors",
        y="Count",
        data=all_distribs,
        palette=colors,
        hue="annotation",
        hue_order=["train", "val"],
    )
    sns.move_legend(
        g, "lower center", bbox_to_anchor=(0.5, 1), ncol=4, title=None, frameon=False
    )

    sns.despine()
    plt.yscale("log")
    # sns.set_palette(sns.color_palette(colors))
    # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    plt.gca().yaxis.set_ticks([1, 10, 100, 1000, 10000])
    plt.gca().set_xlabel("")
    plt.gca().set_axisbelow(True)
    plt.grid(axis="y", which="major", linewidth=1, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", transparent=True)
    plt.close()


def plot_spearman_corr_heatmap(
    spearman, pval, spearman_threshold=0.0, pval_threshold=0.01, fname=None
):

    mask_factor_metaclass = get_mask_from_corr_matrix(
        pval.loc[FACTORS, METACLASSES].values,
        spearman.loc[FACTORS, METACLASSES].values,
        corr_threshold=spearman_threshold,
        pval_threshold=pval_threshold,
    )
    mask_factor_factor = get_mask_from_corr_matrix(
        pval.loc[FACTORS, FACTORS].values,
        spearman.loc[FACTORS, FACTORS].values,
        corr_threshold=spearman_threshold,
        pval_threshold=pval_threshold,
    )
    mask_factor_factor[np.triu_indices_from(mask_factor_factor)] = True

    factors_row_to_keep_idx = np.where(
        ~(mask_factor_metaclass.prod(axis=1) * mask_factor_factor.prod(axis=1)).astype(
            bool
        )
    )[0]
    factors_col_to_keep_idx = np.where(~mask_factor_factor.prod(axis=0).astype(bool))[0]
    metaclasses_col_to_keep_idx = np.where(
        ~mask_factor_metaclass.prod(axis=0).astype(bool)
    )[0]
    factors_row_to_keep = [FACTORS[idx] for idx in factors_row_to_keep_idx]
    factors_col_to_keep = [FACTORS[idx] for idx in factors_col_to_keep_idx]
    metaclasses_col_to_keep = [METACLASSES[idx] for idx in metaclasses_col_to_keep_idx]

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(
            0.6 * (len(metaclasses_col_to_keep) + len(factors_col_to_keep)),
            0.4 * len(factors_row_to_keep),
        ),
        gridspec_kw={
            "width_ratios": [
                len(metaclasses_col_to_keep),
                1.2 * len(factors_col_to_keep),
            ]
        },
        sharey=True,
    )

    g = sns.heatmap(
        100 * spearman.loc[factors_row_to_keep, metaclasses_col_to_keep],
        annot=True,
        mask=mask_factor_metaclass[factors_row_to_keep_idx, :][
            :, metaclasses_col_to_keep_idx
        ],
        ax=axs[0],
        fmt=".1f",
        cmap="RdBu",
        cbar=False,
        vmin=-50,
        vmax=50,
    )

    g = sns.heatmap(
        100 * spearman.loc[factors_row_to_keep, factors_col_to_keep],
        annot=True,
        mask=mask_factor_factor[factors_row_to_keep_idx, :][:, factors_col_to_keep_idx],
        ax=axs[1],
        fmt=".1f",
        cmap="RdBu",
        cbar=True,
        vmin=-50,
        vmax=50,
    )
    axs[1].tick_params(axis="y", left=True, labelleft=False)
    fig.autofmt_xdate(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=300, transparent=True)
    plt.close()


def get_mask_from_corr_matrix(pval, corr, corr_threshold=0.05, pval_threshold=0.01):
    mask = np.ones_like(corr)
    # ind_low = np.unravel_index(np.argsort(corr, axis=None)[:top_k], corr.shape)
    # ind_high = np.unravel_index(np.argsort(corr, axis=None)[-top_k:], corr.shape)
    mask[np.abs(corr) > corr_threshold] = False
    mask[(pval > pval_threshold)] = True
    return mask


def plot_unity(xdata, ydata, **kwargs):
    mn = max(xdata.max(), ydata.max())
    mx = max(xdata.min(), ydata.min())
    points = np.linspace(mn, mx, 100)
    plt.gca().plot(
        points, points, color="k", marker=None, linestyle="--", linewidth=1.0, zorder=-1
    )


def plot_scatter_reg(
    data,
    x="top_1_acc",
    y="acc",
    color=None,
    label=None,
    identity_line=False,
    zero_line=False,
    annotations=None,
    **kwargs,
):

    ax = plt.gca()
    sns.scatterplot(
        data=data[data.model_type != "Original validation annotations"],
        x=x,
        y=y,
        **kwargs,
    )
    # sns.regplot(data=data[data.model_type != "Original validation annotations"], x=x, y=y, scatter=False, order=3, ax=ax, line_kws=dict(zorder=-1), color="#2a9d8f", ci=None, label=None)

    kwargs.update(dict(marker="p", s=80, alpha=1, linewidth=1, edgecolor="w"))
    sns.scatterplot(
        data=data[data.model_type == "Original validation annotations"],
        x=x,
        y=y,
        zorder=2,
        **kwargs,
    )
    deltay = 0.6 if y == "Error ratio" else 0.2
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.20))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(deltay))
    plt.grid(True, linestyle="--", linewidth=1, alpha=0.1, zorder=-1)

    ax.set_axisbelow(True)
    if y == "Error ratio":
        plt.axhline(y=1, color="k", linestyle="--", linewidth=2.0, zorder=-1)
    else:
        plot_unity(data[x], data[y])


def extract_and_filter_by_model_type(df, model_types):
    df["model_type"] = df.index.map(
        lambda x: model_types_map[x].value
        if x in model_types_map
        else "Original validation annotations"
        if x == "human"
        else "Other"
    )
    df = df[df["model_type"].isin(model_types)]
    return df

def set_color_palette():
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    sns.set_palette(sns.color_palette(colors))

def factor_scatterplot(
    df,
    id_vars,
    var_name,
    value_name,
    x,
    y="Error ratio",
    hue=None,
    order=None,
    fname=None,
    remove_underscore=True,
):
    melted_accs, factor_order = melt_table(df, id_vars, var_name, value_name, remove_underscore=remove_underscore)
    melted_accs["Error ratio"] = (1 - melted_accs[value_name]) / (1 - melted_accs[x])
    melted_accs = melted_accs.dropna()

    factor_order = (
        melted_accs.sort_values(by=x, ascending=False)
        .groupby(var_name)
        .head(20)
        .groupby(var_name)[y]
        .mean()
        .sort_values()
        .index
    )
    melted_accs = (
        melted_accs.reset_index()
        .set_index("Factor")
        .loc[factor_order]
        .reset_index()
        .set_index("model")
    )

    set_color_palette()    

    facet = sns.FacetGrid(
        melted_accs,
        col=var_name,
        col_order=factor_order,
        col_wrap=6,
        height=2.3,
        aspect=0.85,
        sharex=True,
        sharey=True,
        margin_titles=True,
        legend_out=False,
    )
    facet.map_dataframe(
        plot_scatter_reg,
        x=x,
        y=y,
        s=20,
        hue=hue,
        hue_order=order,
        style=hue,
        style_order=order,
        alpha=0.6,
        linewidth=0,
        identity_line=True,
    )
    facet.set_titles(col_template="{col_name}")
    facet.add_legend()
    sns.move_legend(
        facet,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=4,
        title=None,
        frameon=False,
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    plt.savefig(fname, bbox_inches="tight", dpi=300, transparent=True)
    plt.close()


def melt_table(df, id_vars, var_name, value_name, remove_underscore=True):
    melted_accs = df.reset_index().melt(
        id_vars=id_vars,
        var_name=var_name,
        value_name=value_name,
    )
    factor_order = melted_accs.groupby(var_name)[value_name].mean().sort_values().index
    melted_accs = (
        melted_accs.set_index(var_name)
        .loc[factor_order]
        .reset_index()
        .set_index("model")
    )
    if remove_underscore:
        melted_accs.replace(r"_", " ", regex=True, inplace=True)
    return melted_accs, factor_order


def load_all_annotations(args):
    # Concatenate all annotations into a single dataframe.
    annotation_dict = {}
    for partition in args.partitions:
        for which_factor in args.which_factor:
            annotations = load_annotations(
                which_factor=which_factor,
                partition=partition,
                filter_prototypes=True,
            )
            annotation_dict[partition, which_factor] = annotations
    return annotation_dict


def linear_regression_coef(x, y, deg=1):
    return np.polyfit(x, y, deg=deg)[-2]


def augmentation_effect(df, augmentation, measure):
    p = np.polyfit(df[augmentation], df[measure], deg=3)

    def poly(p, x):
        return p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]

    x = df[augmentation].unique()
    x.sort()
    groups = df.groupby(augmentation)
    value = poly(p, x)
    max_augment = value.argmax()
    min_augment = value.argmin()
    pop_max = groups.get_group(x[max_augment])[measure].values
    pop_min = groups.get_group(x[min_augment])[measure].values

    # effect = (value[max_augment] - value[min_augment])/value.mean()
    r = stats.ttest_ind(
        pop_min if min_augment < max_augment else pop_max,
        pop_max if min_augment < max_augment else pop_min,
        equal_var=False,
        alternative="two-sided",
    )
    effect, pval = r.statistic, r.pvalue
    return pd.Series({"effect": effect, "pval": pval})


def augmentation_effect_scatter_plot_reduced(
    all_accuracies, all_hparams, augmentation_strengths, fname, y_title="Error ratio", remove_underscore=True,
):
    colors = ["#2a9d8f", "#f4a261"]
    height = 2.2
    aspect = 1.2
    fig, axs = plt.subplots(
        len(all_accuracies),
        6,
        figsize=(height * aspect * 6, height * len(all_accuracies)),
        sharex=True,
    )
    for i, (accuracies, hparams, augmentation_strength) in enumerate(
        zip(all_accuracies, all_hparams, augmentation_strengths)
    ):
        significant = accuracies.groupby(augmentation_strength).size() > 4
        significant = significant[significant].index
        accuracies = accuracies[accuracies[augmentation_strength].isin(significant)]

        melted_accs, factor_order = melt_table(
            accuracies,
            ["model", "average", "worst_factor", "worst_100_classes"] + hparams,
            "Factor",
            y_title,
            remove_underscore=remove_underscore,
        )
        melted_accs[y_title] = (1 - melted_accs[y_title]) / (1 - melted_accs["average"])

        melted_accs = melted_accs.dropna(axis=0, subset=["Error ratio"])

        aug_effect = (
            melted_accs[melted_accs.index.str.lower() != "human"]
            .groupby("Factor")
            .apply(augmentation_effect, augmentation_strength, "Error ratio")
        )
        factor_order_high = (
            aug_effect[aug_effect.pval < 0.05]
            .effect.sort_values(ascending=False)
            .head(3)
            .index
        )
        factor_order_low = (
            aug_effect[aug_effect.pval < 0.05]
            .effect.sort_values(ascending=False)
            .tail(3)
            .index
        )
        factor_order = factor_order_high.append(factor_order_low)

        melted_accs = (
            melted_accs.reset_index()
            .set_index("Factor")
            .loc[factor_order]
            .reset_index()
        )
        groups = melted_accs.groupby("Factor")
        for j, factor in enumerate(factor_order):
            df = groups.get_group(factor)
            color = colors[0] if j < 3 else colors[1]
            g = sns.regplot(
                data=df,
                x=augmentation_strength,
                y="Error ratio",
                scatter=True,
                fit_reg=True,
                line_kws=dict(color=color),
                scatter_kws={"alpha": 0.05, "s": 3, "color": color},
                order=3,
                ax=axs[i, j],
            )
            axs[i, j].annotate(
                factor,
                color=color,
                size=13,
                xy=(0.5, 0.92),
                ha="center",
                xycoords="axes fraction",
            )
            if j > 0:
                axs[i, j].set_ylabel("")
            # axs[i, j].yaxis.set_major_formatter(
            #     ticker.PercentFormatter(xmax=1, decimals=0)
            # )
            axs[i, j].set_xlim(0, 100)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    sns.despine()
    plt.savefig(fname, bbox_inches="tight", dpi=100, transparent=True)
    plt.close()


def extract_hparams_color(accuracies):
    hparams = ["crop", "wd", "id", "Color strength"]
    for hparam in hparams:
        name = "color" if hparam == "Color strength" else hparam
        accuracies[hparam] = pd.to_numeric(
            accuracies.index.to_series().str.extract(
                f"{name}"+r"((?:[-?[\d.]+(?:e-?\d+)?)|(?:[+-]?[0-9]*[.]?[0-9]+]))"
            )[0]
        )

    return hparams


def extract_hparams_smoothing(accuracies):
    hparams = ["crop", "wd", "id", "Smoothing strength"]
    for hparam in hparams:
        name = "smoothing" if hparam == "Smoothing strength" else hparam
        accuracies[hparam] = pd.to_numeric(
            accuracies.index.to_series().str.extract(
                f"{name}"+r"((?:[-?[\d.]+(?:e-?\d+)?)|(?:[+-]?[0-9]*[.]?[0-9]+]))"
            )[0]
        )

    return hparams


def extract_hparams_crop(accuracies):
    hparams = ["Crop strength", "wd", "id"]
    for hparam in hparams:
        name = "crop" if hparam == "Crop strength" else hparam
        accuracies[hparam] = pd.to_numeric(
            accuracies.index.to_series().str.extract(f"{name}"+r"([+-]?[0-9]*[.]?[0-9]+)")[
                0
            ]
        )
        if hparam == "Crop strength":
            accuracies[hparam] = 100 - accuracies[hparam]

    return hparams


class ModelRobustnessPlot:
    def __init__(self, prevalence: Prevalence, model: str, top_n_factors: int = 3):
        self.prevalence = prevalence
        self.model = model
        self.top_n_factors = top_n_factors

        models = set(prevalence.metaclass_misclassification_shifts_df.model.unique())
        assert model in models, f"{model} not in {models}"

        # misclassification shifts sorted by accuracy
        self.shifts = prevalence.metaclass_misclassification_shifts_df[
            prevalence.metaclass_misclassification_shifts_df["model"] == model
        ].sort_values(by="accuracy")

    def plot_metaclass(self, metaclass: str = "dog") -> go.Figure:
        fig = go.Figure()
        metaclass_shifts = self.shifts[self.shifts["metaclass"] == metaclass]
        bar = self.make_metaclass_bar(metaclass_shifts)
        fig.add_trace(bar)
        accuracy = metaclass_shifts["accuracy"].item()
        fig.update_layout(title=f"{metaclass} ({accuracy:,.2%} top-1)")
        fig = self.stylize(fig)
        return fig

    def make_metaclass_bar(
        self, metaclass_shifts: pd.DataFrame, show_legend: bool = False
    ) -> go.Bar:
        values = (
            metaclass_shifts[self.prevalence.factors]
            .squeeze()
            .sort_values(ascending=True)
        )
        x, y = values.values, values.index

        # name = "robust" if robust else "susceptible"
        colors = ["green" if s < 0 else "#DC3912" for s in x]

        bar = go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker_color=colors,
            showlegend=show_legend,
        )

    def make_robust_or_susceptible_bar(
        self,
        metaclass_shifts: pd.DataFrame,
        show_legend: bool = False,
        robust: bool = True,
    ) -> go.Bar:
        values = (
            metaclass_shifts[self.prevalence.factors]
            .squeeze()
            .sort_values(ascending=True)
        )
        # keep only robust or susceptible
        x, y = values.values, values.index
        positive_filter = x >= 0
        negative_filter = x < 0

        i_filter = negative_filter if robust else positive_filter
        x, y = x[i_filter], y[i_filter]

        name = "robust" if robust else "susceptible"
        color = "green" if robust else "#DC3912"

        bar = go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker_color=color,
            name=name,
            showlegend=show_legend,
            legendgroup=name,
        )
        return bar

    def plot_worst_3_and_dog(
        self,
        vertical_spacing=0.03,
        horiztonal_spacing=0.15,
    ) -> go.Figure:
        metaclasses = list(self.shifts["metaclass"].values)
        accuracies = [
            self.shifts[self.shifts["metaclass"] == m]["accuracy"].item()
            for m in metaclasses
        ]
        # TODO: dynamically sort worst metaclasses

        fig = self.plot(
            metaclasses=["vessel", "snake", "commodity", "dog"],
            num_cols=4,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horiztonal_spacing,
        )
        return fig

    def plot(
        self,
        metaclasses=None,
        vertical_spacing=0.03,
        horizontal_spacing=0.10,
        num_cols=3,
    ) -> go.Figure:
        if metaclasses is None:
            metaclasses = list(self.shifts["metaclass"].values)

        accuracies = [
            self.shifts[self.shifts["metaclass"] == m]["accuracy"].item()
            for m in metaclasses
        ]

        num_rows = -(len(metaclasses) // -num_cols)

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            subplot_titles=[
                f"{m} <br> <span style='font-size: 10px;'>({a:,.2%} top-1)</span> "
                for m, a in zip(metaclasses, accuracies)
            ],
        )

        for i, metaclass in enumerate(metaclasses):
            show_legend = True if i == 0 else False
            row = i // num_cols
            col = i % num_cols

            metaclass_shifts = self.shifts[self.shifts["metaclass"] == metaclass]
            robust_bars = self.make_robust_or_susceptible_bar(
                metaclass_shifts, show_legend=show_legend, robust=True
            )
            fig.add_trace(robust_bars, row=row + 1, col=col + 1)

            susceptible_bars = self.make_robust_or_susceptible_bar(
                metaclass_shifts, show_legend=show_legend, robust=False
            )
            fig.add_trace(susceptible_bars, row=row + 1, col=col + 1)

        fig.update_layout(barmode="group")
        fig = self.stylize(fig)
        fig.update_layout(title=f"{self.model}", showlegend=True)
        return fig

    def to_pdf(self, save_dir: str = "plots/"):
        fig = self.plot(vertical_spacing=0.03, horizontal_spacing=0.10)
        fig.update_layout(height=2200, width=1500)
        fig.write_image(os.path.join(save_dir, f"{self.model}_metaclass_dashboard.pdf"))

    def worst_to_pdf(self, save_dir: str = "plots/"):
        fig = self.plot_worst_3_and_dog(horiztonal_spacing=0.20)
        fig.update_layout(height=450)
        fig.update_layout(font={"size": 11})
        fig.write_image(os.path.join(save_dir, f"{self.model}_worst_metaclass.pdf"))

    @staticmethod
    def stylize(fig: go.Figure):
        fig.update_layout(template="plotly_white")
        return fig


class SingleModelOverall:
    RED = "#C41E3A"

    def __init__(self, prevalence):
        self.prevalence = prevalence

    def stylize(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            # font_family="Serif",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        # fig.update_layout(
        #     showlegend=True,
        #     legend=dict(xanchor="right", x=1.0, y=1.1),
        #     margin={"r": 10},
        #     height=300,
        # )
        # fig.update_xaxes(tickfont=dict(size=14))
        return fig

    def make_robust_or_susceptible_bar(
        self,
        shifts_values,
        factors,
        show_legend: bool = False,
        robust: bool = True,
    ) -> go.Bar:
        # keep only robust or susceptible
        x, y = shifts_values, factors
        positive_filter = x >= 0
        negative_filter = x < 0

        i_filter = negative_filter if robust else positive_filter
        x, y = x[i_filter], y[i_filter]

        name = "robust" if robust else "susceptible"
        color = "#2a9d8f" if robust else "#f4a261"

        bar = go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker_color=color,
            name=name,
        )
        return bar

    def plot(self, model="ViT") -> go.Figure:
        df = self.prevalence.misclassification_shifts_df
        df = df[df["model"] == model]

        fig = go.Figure()
        susceptible_bars = self.make_robust_or_susceptible_bar(
            df["shift"].values, df["factor"].values, robust=False
        )
        fig.add_trace(susceptible_bars)
        robust_bars = self.make_robust_or_susceptible_bar(
            df["shift"].values, df["factor"].values, robust=True
        )
        fig.add_trace(robust_bars)
        fig = self.stylize(fig)
        return fig


class CompareModelsPlot:
    COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    MODEL_TO_COLORS = dict(zip(["ViT", "ResNet50", "SimCLR", "DINO"], COLORS))
    RED = "#C41E3A"

    def __init__(self, prevalence):
        self.prevalence = prevalence

    def stylize(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            font_family="Serif",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_layout(yaxis={"categoryorder": "total descending"})
        fig.update_layout(
            showlegend=True,
            legend=dict(xanchor="right", x=1.0, y=1.1),
            margin={"r": 10},
            height=300,
        )
        fig.update_xaxes(tickfont=dict(size=14))
        return fig

    def plot(self, models=["ViT", "ResNet50", "SimCLR", "DINO"]) -> go.Figure:
        df = self.prevalence.misclassification_shifts_df
        df = df[df["model"].isin(models)]

        fig = px.bar(
            df,
            x="factor",
            y="shift",
            color_discrete_sequence=[
                "#264653",
                "#2a9d8f",
                "#e9c46a",
                "#f4a261",
                "#e76f51",
            ],
            color="model",
            barmode="group",
            color_discrete_map=self.MODEL_TO_COLORS,
            height=250,
            width=750,
        )
        fig.add_annotation(
            x=-1.0,
            y=0.5,
            text="less robust \u2192",
            showarrow=False,
            align="left",
            font={"color": self.RED, "size": 11},
            textangle=-90,
        )

        fig = self.stylize(fig)
        return fig

    def to_pdf(self, save_dir: str = "plots/"):
        fig = self.plot()
        fig.write_image(os.path.join(save_dir, "model_comparison.pdf"))
        fig = self.plot(["ViT"])
        fig.write_image(os.path.join(save_dir, "vit_bias.pdf"))
        fig = self.plot(["ViT", "ResNet50"])
        fig.write_image(os.path.join(save_dir, "vit_v_resnet50.pdf"))
        fig = self.plot(["ViT", "DINO"])
        fig.write_image(os.path.join(save_dir, "vit_v_dino.pdf"))
        fig = self.plot(["ResNet50", "SimCLR"])
        fig.write_image(os.path.join(save_dir, "resnet50_v_simclr.pdf"))


def model_comparison(
    comparison_df,
    value="Error ratio",
    average_name="average",
    fname=None,
    hue="model",
    hue_order=None,
    show_significance=False,
    compact=False,
    remove_underscore=True
):
    if hue_order is not None:
        comparison_df = comparison_df[comparison_df[hue].isin(hue_order)]
    comparison_df = comparison_df[FACTORS + [hue, average_name]].melt(
        id_vars=[hue, average_name], var_name="Factor", value_name="Accuracy"
    )
    comparison_df["Error ratio"] = (1 - comparison_df["Accuracy"]) / (
        1 - comparison_df[average_name]
    )
    if remove_underscore:
        comparison_df.replace(r"_", " ", regex=True, inplace=True)
    comparison_df = comparison_df.sort_values(by=value)
    factor_order = (
        comparison_df.groupby("Factor")["Error ratio"].mean().sort_values().index.values
    )
    if compact:
        factor_order = list(factor_order[:3]) + list(factor_order[-3:])

    def test_significance(df):
        factors, dfs = zip(*df.groupby(hue))
        values = [df[value].values for df in dfs]
        return stats.alexandergovern(*values).pvalue

    if show_significance:
        p_val = comparison_df.groupby("Factor").apply(test_significance)
        signif = p_val < 0.05
        signif = signif.loc[factor_order]
    
    data_to_display = comparison_df.groupby(["Factor", hue]).mean().reset_index()
    g = plot_bar_plot(data_to_display, x="Factor", y=value, hue=hue, hue_order=hue_order, factor_order=factor_order)
    
    
    if show_significance:
        x = []
        xlabel = []
        for c in g.containers:
            for bar, factor, s in zip(c, factor_order, signif):
                x.append(bar.xy[0] + bar.get_width() / 2)
                xlabel.append((factor, c.get_label()))
                if show_significance and not s:
                    bar.set_alpha(0.3)
        comparison_df["Error ratio"] = comparison_df["Error ratio"]
        y = (
            comparison_df.groupby(["Factor", hue])
            .mean()
            .loc[xlabel]
            .reset_index()[value]
            .values
        )
        yerr = (
            comparison_df.groupby(["Factor", hue])
            .std()
            .loc[xlabel]
            .reset_index()[value]
            .values
        )
        xlim = g.get_xlim()
        plt.errorbar(x, y, yerr=yerr, linewidth=0, elinewidth=1.8, alpha=1, c=".35")
        plt.xlim(xlim)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()

def plot_bar_plot(data_to_display, x="Factor", y=None, hue=None, hue_order=None, factor_order=None, compact=False):
    data_to_display = data_to_display.copy()
    if y == "Error ratio":
        data_to_display[y] = data_to_display[y] - 1
    plt.figure(figsize=(3.5 if compact else 10, 4.5 if compact else 2.8))
    g = sns.barplot(
        data=data_to_display,
        x=x,
        y=y,
        order=factor_order,
        hue=hue,
        hue_order=hue_order,
        bottom=1 if y == "Error ratio" else 0,
    )
    
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("")
    plt.ylabel(y, fontsize=18)
    sns.despine()
    # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    if y == "Error ratio":
        # annotate less robust direction
        g.annotate(
            "less robust \u2192",
            xy=(-0.1, 1),
            xytext=(-0.1, 1.05),
            xycoords="data",
            textcoords="data",
            ha="center",
            va="bottom",
            rotation=90,
            size=11,
            color="#e76f51",
        )
        g.axhline(y=1, color="k", alpha=0.8, linestyle="-", linewidth=1.5)
    if hue:
        sns.move_legend(
            g,
            "lower left",
            bbox_to_anchor=(-0.05 if compact else 0.01, 1.01 if compact else 0.78),
            ncol=1 if compact else 2,
            title=None,
            frameon=False,
        )
    plt.tight_layout()
    plt.gca().set_axisbelow(True)
    plt.grid(axis="y", which="major", linewidth=1, alpha=0.3, linestyle="--")
    return g


def color_coded_latex_table(accuracies, fname):
    error_ratios = (1 - accuracies[FACTORS]).divide(1 - accuracies["average"], axis=0)
    error_ratios = (
        error_ratios.reset_index()
        .melt(
            id_vars=["model", "metaclass"], var_name="Factor", value_name="Error ratio"
        )
        .dropna()
        .sort_values("Error ratio")
    )
    error_ratios.metaclass = error_ratios.metaclass.str.replace("_", " ")

    def format_(df, positive=True):
        c = pd.Series(np.where(df["Error ratio"] <= 1, colors[1], colors[3]))
        df = df[df["Error ratio"] <= 1 if positive else df["Error ratio"] > 1]
        annot = (
            df["Factor"].str.replace("_", " ")
            + " ("
            + df["Error ratio"].apply(r"{:.1f}".format)
            + ")"
        ).values
        annot = ", ".join(list(annot[:4]) if positive else list(annot[-4:]))
        return r"{\color[HTML]{" + colors[1 if positive else 3][1:] + "}" + annot+"}"

    positive_error_ratios = (
        error_ratios.groupby(["model", "metaclass"])
        .apply(format_, positive=True)
        .to_frame("Robust")
    )
    negative_error_ratios = (
        error_ratios.groupby(["model", "metaclass"])
        .apply(format_, positive=False)
        .to_frame("Susceptible")
    )
    df = pd.concat([positive_error_ratios, negative_error_ratios], axis=1)

    df.columns = [
        "\\textbf{" + x.replace("_", " ") + "}" for x in df.columns
    ]
    df.index.names = ["\\textbf{" + x + "}" for x in df.index.names]
    df.style.to_latex(
        fname, hrules=True, column_format="llll"
    )


def specific_model_metaclasses(
    accuracies_default_models, accuracies_default_models_metaclass, fname, remove_underscore=True
):
    accuracies_vit_metaclass = accuracies_default_models_metaclass[
        accuracies_default_models_metaclass.model == "ViT"
    ]
    accuracies_vit_metaclass = accuracies_vit_metaclass.sort_values("average")
    metaclasses_to_display = accuracies_vit_metaclass.metaclass.head(3).values
    metaclasses_to_display = list(metaclasses_to_display) + ["dog"]
    accuracies_vit_metaclass = accuracies_vit_metaclass[
        FACTORS + ["average", "metaclass"]
    ].melt(id_vars=["average", "metaclass"], var_name="Factor", value_name="Accuracy")
    accuracies_vit_metaclass["Error ratio"] = (
        1 - accuracies_vit_metaclass["Accuracy"]
    ) / (1 - accuracies_vit_metaclass["average"])
    accuracies_vit_metaclass = accuracies_vit_metaclass.replace(r"_", " ", regex=True)
    accuracies_vit_metaclass = accuracies_vit_metaclass.dropna()

    fig, axs = plt.subplots(1, 5, figsize=(12, 2.2), sharex=True)
    # plt.subplots_adjust(hspace=0.3, wspace=-0.1)
    sns.despine()
    groups = accuracies_vit_metaclass.groupby("metaclass")
    for i, metaclass in enumerate(metaclasses_to_display):
        df = groups.get_group(metaclass)
        df = df.sort_values("Error ratio")
        df = pd.concat([df.head(3), df.tail(3)])
        axs[i].barh(
            df.head(3)["Factor"], df.head(3)["Error ratio"] - 1, color=colors[1], left=1
        )
        axs[i].barh(
            df.tail(3)["Factor"], df.tail(3)["Error ratio"] - 1, color=colors[3], left=1
        )
        axs[i].set_title(metaclass)
        axs[i].set_axisbelow(True)
        axs[i].grid(axis="x", which="major", linewidth=1, alpha=0.3, linestyle="--")
        axs[i].axvline(x=1, color="k", alpha=0.8, linestyle="-", linewidth=1.5)
        axs[i].set_xlabel("Error ratio", fontsize=14)
        # plt.setp(axs[i].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    overall_vit = accuracies_default_models.loc[["ViT"]]
    overall_vit = overall_vit[FACTORS + ["average"]].melt(
        id_vars=["average"], var_name="Factor", value_name="Accuracy"
    )
    overall_vit["Error ratio"] = (1 - overall_vit["Accuracy"]) / (
        1 - overall_vit["average"]
    )
    if remove_underscore:
        overall_vit.replace(r"_", " ", regex=True, inplace=True)
    overall_vit = overall_vit.sort_values("Error ratio")
    overall_vit = pd.concat([overall_vit.head(3), overall_vit.tail(3)])
    axs[4].barh(
        overall_vit.head(3)["Factor"],
        overall_vit.head(3)["Error ratio"] - 1,
        color=colors[1],
        label="Robust",
        left=1,
    )
    axs[4].barh(
        overall_vit.tail(3)["Factor"],
        overall_vit.tail(3)["Error ratio"] - 1,
        color=colors[3],
        label="Susceptible",
        left=1,
    )
    axs[4].set_title("overall")
    axs[4].set_axisbelow(True)
    axs[4].grid(axis="x", which="major", linewidth=1, alpha=0.3, linestyle="--")
    axs[4].axvline(x=1, color="k", alpha=0.8, linestyle="-", linewidth=1.5)
    axs[4].set_xlabel("Error ratio", fontsize=14)
    fig.legend()
    sns.move_legend(
        fig,
        "lower center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=2,
        title=None,
        frameon=False,
    )
    # plt.setp(axs[4].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", transparent=True)
    plt.close()


def get_augmentation_comparison_df(
    accuracies_crop,
    accuracies_color,
    accuracies_color_crop,
    accuracies_color_crop_adaptive,
):
    comparison = {
        "Crop": accuracies_crop[
            accuracies_crop["Crop strength"] == accuracies_crop["Crop strength"].max()
        ].copy(),
        "Color": accuracies_color[
            accuracies_color["Color strength"]
            == accuracies_color["Color strength"].max()
        ].copy(),
        "Full Color+Crop": accuracies_color_crop.copy(),
        "Per class color jitter + crop": accuracies_color_crop_adaptive.copy(),
    }
    for key, value in comparison.items():
        value["Augmentation type"] = key
    comparison_df = pd.concat(comparison.values())
    comparison_df = comparison_df.drop(["human"], axis=0, errors="ignore")
    return comparison, comparison_df


def generate_all_plots(args):
    plot_dir = Path(args.plot_dir)
    results_dir = Path(args.results_dir)
    os.makedirs(plot_dir, exist_ok=True)

    if args.use_tex:
        plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 12})
        plt.rc("text", usetex=True)

    annotation_dict = load_all_annotations(args)

    factor_distribution_comparison(
        annotation_dict,
        fname=plot_dir / f"factor_distribution_comparison.{args.format}",
    )

    active_factor_distribution_comparison(
        annotation_dict,
        fname=plot_dir / f"active_factor_distribution_comparison.{args.format}",
    )

    # Compute the correlation between the factors and the metaclass.
    spearman_val, pval_val = spearman_corr_heatmaps(
        annotation_dict["val", "multi"], FACTORS, METACLASSES
    )

    spearman_train, pval_train = spearman_corr_heatmaps(
        annotation_dict["train", "multi"], FACTORS, METACLASSES
    )

    plot_spearman_corr_heatmap(
        spearman_val,
        pval_val,
        fname=plot_dir / f"spearman_val_correlation_heatmap.{args.format}",
    )

    plot_spearman_corr_heatmap(
        spearman_train,
        pval_train,
        fname=plot_dir / f"spearman_train_correlation_heatmap.{args.format}",
    )

    for which_factor in args.which_factor:
        for error_type in args.error_types:
            order = (
                MODEL_TYPE_ORDER + ["Original validation annotations"]
                if error_type == "real_class"
                else MODEL_TYPE_ORDER
            )

            plot_dir = Path(args.plot_dir) / which_factor / error_type
            os.makedirs(plot_dir, exist_ok=True)

            accuracies_testbed = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/imagenet_testbed/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")
            accuracies_model_vs_human = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/modelvshuman/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")
            accuracies_default_models = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/base/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")
            accuracies_model_vs_human = accuracies_model_vs_human[
                accuracies_model_vs_human.index.isin(SELFSUPERVISED_MODELS)
            ]
            accuracies_default_models_ssl = accuracies_default_models[
                accuracies_default_models.index.isin(SELFSUPERVISED_MODELS)
            ]
            accuracies = pd.concat(
                [
                    accuracies_testbed,
                    accuracies_model_vs_human,
                    accuracies_default_models_ssl,
                ]
            )

            accuracies = extract_and_filter_by_model_type(accuracies, order)
            x_title = {
                "class": "Top 1 Accuracy",
                "real_class": "ReaL Accuracy",
                "metaclass": "Metaclass Accuracy",
            }[error_type]
            accuracies.rename(columns={"average": x_title}, inplace=True)
            y_title = f"{x_title} (Factor)"

            print(
                f"Showing {len(accuracies)} models for main scatter plot for {which_factor} factor and {error_type} error type"
            )
            factor_scatterplot(
                accuracies,
                id_vars=["model", x_title, "model_type"],
                var_name="Factor",
                value_name=y_title,
                x=x_title,
                y=y_title,
                hue="model_type",
                order=order,
                fname=plot_dir / f"highlevel_scatter_plot.{args.format}",
            )

            hue_order = MODEL_TYPE_ORDER.copy()

            model_comparison(
                accuracies,
                "Error ratio",
                hue="model_type",
                average_name=x_title,
                hue_order=hue_order,
                show_significance=True,
                fname=plot_dir / f"supervision_comparison_prevalence.{args.format}",
            )

            accuracies_default_models_metaclasses = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/base/{error_type}/accuracies_per_factor_metaclass.csv"
            )

            color_coded_latex_table(
                accuracies_default_models_metaclasses.set_index(
                    ["model", "metaclass"]
                ).sort_index(),
                fname=plot_dir / f"accuracies_per_factor_metaclass.tex",
            )

            specific_model_metaclasses(
                accuracies_default_models.copy(),
                accuracies_default_models_metaclasses,
                fname=plot_dir / f"vit_example.{args.format}",
            )

            accuracies_crop = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/crop/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")

            hparams_crop = extract_hparams_crop(accuracies_crop)

            significant = accuracies_crop.groupby("Crop strength").size() > 4
            significant = significant[significant].index
            accuracies_crop = accuracies_crop[
                accuracies_crop["Crop strength"].isin(significant)
            ]

            accuracies_color = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/colorjitter/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")

            hparams_color = extract_hparams_color(accuracies_color)

            accuracies_smoothing = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/gaussian_blur/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")

            hparams_smoothing = extract_hparams_smoothing(accuracies_smoothing)

            augmentation_effect_scatter_plot_reduced(
                [accuracies_crop, accuracies_color, accuracies_smoothing],
                [hparams_crop, hparams_color, hparams_smoothing],
                ["Crop strength", "Color strength", "Smoothing strength"],
                fname=plot_dir / f"augment_effect.{args.format}",
            )

            accuracies_color_crop = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/colorjitter_crop/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")

            accuracies_color_crop_adaptive = pd.read_csv(
                results_dir / f"val/{which_factor}_factor/adaptive_colorjitter/{error_type}/accuracies_per_factor.csv"
            ).set_index("model")

            comparison, comparison_df = get_augmentation_comparison_df(
                accuracies_crop,
                accuracies_color,
                accuracies_color_crop,
                accuracies_color_crop_adaptive,
            )

            model_comparison(
                comparison_df,
                "Error ratio",
                hue="Augmentation type",
                hue_order=comparison.keys(),
                show_significance=True,
                fname=plot_dir / f"augment_comparison_prevalence.{args.format}",
            )
            model_comparison(
                comparison_df,
                "Accuracy",
                hue="Augmentation type",
                hue_order=comparison.keys(),
                show_significance=True,
                fname=plot_dir / f"augment_comparison_accuracy.{args.format}",
            )

def add_plot_args(parser):
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--error-types",
        nargs="+",
        default=["class", "real_class", "metaclass"],
        choices=["real_class", "metaclass", "class"],
    )
    parser.add_argument("--use-tex", action="store_true")
    parser.add_argument("--format", default="pdf", choices=["png", "pdf"])
    parser.add_argument(
        "--which-factor", nargs="+", default=["top", "multi"], choices=["multi", "top"]
    )
    parser.add_argument(
        "--partitions", nargs="+", default=["train", "val"], choices=["train", "val"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_plot_args(parser)
    args = parser.parse_args()

    generate_all_plots(args)
