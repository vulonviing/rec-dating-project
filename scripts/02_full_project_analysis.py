from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_dating_project import (
    PopularityPrestigeAnalyzer,
    ProjectPaths,
    RecDatingDataset,
    RoleBasedBipartiteNetwork,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full rec-dating project analysis.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional number of rows to read.")
    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=8,
        help="Minimum rating used for the positive-layer analysis.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="How many top rows to export.")
    return parser.parse_args()


def run_label(nrows: int | None) -> str:
    return "full" if nrows is None else str(nrows)

def ccdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    array.sort()
    n = array.size
    if n == 0:
        return np.array([]), np.array([])
    y = 1.0 - (np.arange(1, n + 1) - 1) / n
    return array, y


def lorenz_curve(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array >= 0)]
    if array.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    sorted_array = np.sort(array)
    cumulative = np.cumsum(sorted_array)
    cumulative = np.insert(cumulative, 0, 0.0)
    if cumulative[-1] == 0:
        y = np.linspace(0, 1, cumulative.size)
    else:
        y = cumulative / cumulative[-1]
    x = np.linspace(0, 1, cumulative.size)
    return x, y


def popularity_prestige_binned_summary(
    frame: pd.DataFrame,
    num_groups: int = 25,
) -> pd.DataFrame:
    ranked = frame[["popularity_rank_pct", "prestige_rank_pct"]].sort_values(
        "popularity_rank_pct"
    ).reset_index(drop=True)
    ranked["pop_group"] = np.floor(np.arange(len(ranked)) * num_groups / len(ranked)).astype(int)
    summary = ranked.groupby("pop_group", observed=False).agg(
        popularity_mid=("popularity_rank_pct", "median"),
        prestige_median=("prestige_rank_pct", "median"),
        prestige_q25=("prestige_rank_pct", lambda s: s.quantile(0.25)),
        prestige_q75=("prestige_rank_pct", lambda s: s.quantile(0.75)),
    )
    return summary.reset_index(drop=True)

def main() -> None:
    args = parse_args()

    paths = ProjectPaths.default()
    paths.ensure_output_dirs()
    figures_dir = paths.output_figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    label = run_label(args.nrows)

    dataset = RecDatingDataset(paths.raw_edges_path)
    network = RoleBasedBipartiteNetwork(dataset)

    dataset_summary = dataset.compute_summary(nrows=args.nrows)
    full_snapshot = network.build_sparse_rating_matrix(
        nrows=args.nrows,
        summary=dataset_summary,
    )
    full_analyzer = PopularityPrestigeAnalyzer(full_snapshot)
    full_analyzer.compute_hits(max_iter=200, tol=1e-7)
    profile_metrics = full_analyzer.profile_metrics()
    rater_metrics = full_analyzer.rater_metrics()

    positive_summary = dataset.compute_summary(
        nrows=args.nrows,
        min_rating=args.positive_threshold,
    )
    positive_snapshot = network.build_sparse_rating_matrix(
        nrows=args.nrows,
        summary=positive_summary,
        min_rating=args.positive_threshold,
    )
    positive_analyzer = PopularityPrestigeAnalyzer(positive_snapshot)
    positive_profile_metrics = positive_analyzer.profile_metrics()

    top_profiles_popularity = profile_metrics.sort_values("in_strength", ascending=False).head(args.top_k)
    top_profiles_prestige = profile_metrics.sort_values("authority_score", ascending=False).head(args.top_k)
    top_profiles_gap_positive = profile_metrics.sort_values("prestige_gap", ascending=False).head(args.top_k)
    top_profiles_gap_negative = profile_metrics.sort_values("prestige_gap", ascending=True).head(args.top_k)
    top_raters_activity = rater_metrics.sort_values("out_strength", ascending=False).head(args.top_k)
    top_raters_hub = rater_metrics.sort_values("hub_score", ascending=False).head(args.top_k)
    full_correlations = full_analyzer.popularity_vs_prestige_correlation(profile_metrics)

    rating_dist = dataset_summary.rating_distribution_frame()

    profile_metrics.to_csv(paths.output_data_dir / f"profile_metrics_{label}.csv", index=False)
    rater_metrics.to_csv(paths.output_data_dir / f"rater_metrics_{label}.csv", index=False)
    positive_profile_metrics.to_csv(paths.output_data_dir / f"profile_metrics_positive_{args.positive_threshold}_{label}.csv", index=False)
    top_profiles_popularity.to_csv(paths.output_data_dir / f"top_profiles_popularity_{label}.csv", index=False)
    top_profiles_prestige.to_csv(paths.output_data_dir / f"top_profiles_prestige_{label}.csv", index=False)
    top_profiles_gap_positive.to_csv(paths.output_data_dir / f"top_profiles_prestige_gap_positive_{label}.csv", index=False)
    top_profiles_gap_negative.to_csv(paths.output_data_dir / f"top_profiles_prestige_gap_negative_{label}.csv", index=False)
    top_raters_activity.to_csv(paths.output_data_dir / f"top_raters_activity_{label}.csv", index=False)
    top_raters_hub.to_csv(paths.output_data_dir / f"top_raters_hub_{label}.csv", index=False)
    rating_dist.to_csv(paths.output_data_dir / f"rating_distribution_{label}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(rating_dist["rating"], rating_dist["share"], color="#386641")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Share of Edges")
    plt.tight_layout()
    plt.savefig(figures_dir / f"rating_distribution_{label}.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x_profile, y_profile = ccdf(profile_metrics["in_strength"])
    x_rater, y_rater = ccdf(rater_metrics["out_strength"])
    axes[0].plot(x_profile, y_profile, color="#1d3557")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Profile In-Strength CCDF")
    axes[0].set_xlabel("In-Strength")
    axes[0].set_ylabel("CCDF")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_rater, y_rater, color="#e63946")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Rater Out-Strength CCDF")
    axes[1].set_xlabel("Out-Strength")
    axes[1].set_ylabel("CCDF")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"strength_ccdf_{label}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        profile_metrics["in_strength"],
        profile_metrics["authority_score"],
        s=6,
        alpha=0.2,
        color="#ff006e",
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Popularity vs Prestige")
    ax.set_xlabel("Profile Popularity (In-Strength)")
    ax.set_ylabel("Profile Prestige (Authority)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"popularity_vs_prestige_{label}.png", dpi=220)
    plt.close(fig)

    binned = popularity_prestige_binned_summary(profile_metrics, num_groups=25)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.fill_between(
        binned["popularity_mid"] * 100,
        binned["prestige_q25"] * 100,
        binned["prestige_q75"] * 100,
        color="#c7dff2",
        alpha=0.95,
        label="Interquartile range",
    )
    ax.plot(
        binned["popularity_mid"] * 100,
        binned["prestige_median"] * 100,
        color="#1f4f8a",
        linewidth=3.0,
        label="Median prestige percentile",
    )
    ax.plot([0, 100], [0, 100], linestyle="--", color="0.5", linewidth=1.3, label="45° reference")
    ax.set_title("Popularity and Prestige Alignment by Percentile")
    ax.set_xlabel("Popularity percentile")
    ax.set_ylabel("Prestige percentile")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, loc="upper left")
    ax.text(
        97,
        7,
        "Profiles grouped into 25\npopularity quantiles",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    plt.tight_layout()
    plt.savefig(figures_dir / f"popularity_vs_prestige_2_{label}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x1, y1 = lorenz_curve(profile_metrics["in_strength"])
    x2, y2 = lorenz_curve(profile_metrics["authority_score"])
    ax.plot(x1, y1, label="In-Strength", color="#457b9d")
    ax.plot(x2, y2, label="Authority", color="#e76f51")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_title("Lorenz Curves for Profile Attention and Prestige")
    ax.set_xlabel("Cumulative Share of Profiles")
    ax.set_ylabel("Cumulative Share of Value")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"profile_lorenz_{label}.png", dpi=220)
    plt.close(fig)

    print(f"Saved full analysis outputs to {paths.outputs_dir}")
    print(json.dumps(full_correlations, indent=2))


if __name__ == "__main__":
    main()
