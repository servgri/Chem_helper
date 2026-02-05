"""Metric comparison plots for Tox21 model benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PathLike = Union[str, Path]


def _save(fig: plt.Figure, path: Optional[PathLike]) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)


def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    title: Optional[str] = None,
    path: Optional[PathLike] = None,
    vmin: float = 0.5,
    vmax: float = 1.0,
    cmap: str = "YlGnBu",
) -> plt.Figure:
    pivot = metrics_df.pivot_table(index="model_name", columns="target", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, max(4, 0.45 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
    ax.set_title(title or metric)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_mean_metrics_bars(
    metrics_df: pd.DataFrame,
    metrics: Optional[list] = None,
    path: Optional[PathLike] = None,
) -> plt.Figure:
    metrics = metrics or ["test_roc_auc", "test_pr_auc", "test_f1", "test_balanced_acc"]
    mean_df = metrics_df.groupby("model_name")[metrics].mean().sort_values(metrics[0], ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_df.plot(kind="bar", ax=ax)
    ax.set_title("Средние метрики по всем assay")
    ax.set_ylabel("score")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_nr_vs_sr(
    metrics_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    path: Optional[PathLike] = None,
) -> plt.Figure:
    fam = metrics_df.groupby(["family", "model_name"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=fam, x="model_name", y=metric, hue="family", ax=ax, palette=["#2a6f97", "#bc4749"])
    ax.set_title(f"NR vs SR: {metric}")
    ax.set_ylim(0.5, 1.0)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_classical_vs_nn(
    metrics_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    path: Optional[PathLike] = None,
) -> plt.Figure:
    df = metrics_df.copy()
    nn_names = {
        "Lightning_mlp",
        "Lightning_resnet",
        "Lightning_transformer",
        "Lightning_gnn",
        "LightningMLP",
        "LightningResNet",
        "LightningTransformer",
        "LightningGNN",
        "MLP",
        "ResNet",
        "Transformer",
        "GNN",
    }
    df["group"] = df["model_name"].apply(
        lambda m: "Neural (Lightning)"
        if m in nn_names or str(m).startswith("Lightning")
        else "Classical (FLAML)"
    )
    g = df.groupby(["group", "model_name"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=g, x="model_name", y=metric, hue="group", ax=ax, dodge=False)
    ax.set_title(f"Classical FLAML vs Lightning NN — {metric}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_model_ranking_lines(
    metrics_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    path: Optional[PathLike] = None,
) -> plt.Figure:
    """Line chart: each model across endpoints (visual comparison strip)."""
    pivot = metrics_df.pivot_table(index="target", columns="model_name", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", linewidth=1.5, label=col)
    ax.set_title(f"Сравнение моделей по endpoint'ам ({metric})")
    ax.set_ylabel(metric)
    ax.set_ylim(0.45, 1.02)
    plt.xticks(rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_runtime_bars(
    metrics_df: pd.DataFrame,
    path: Optional[PathLike] = None,
) -> Optional[plt.Figure]:
    if "elapsed_sec" not in metrics_df.columns:
        return None
    t = metrics_df.groupby("model_name")["elapsed_sec"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    t.plot(kind="barh", ax=ax, color="#6c757d")
    ax.set_xlabel("mean seconds")
    ax.set_title("Среднее время поиска/обучения")
    fig.tight_layout()
    _save(fig, path)
    return fig
