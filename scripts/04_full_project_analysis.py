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


def build_rank_columns(frame: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    ranked = frame.copy()
    ranked[f"{prefix}_rank"] = (
        ranked[value_col].rank(ascending=False, method="min").astype(int)
    )
    return ranked


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


def top_overlap(frame: pd.DataFrame, col_a: str, col_b: str, k: int) -> dict[str, float]:
    a = set(frame.sort_values(col_a, ascending=False).head(k)["profile_id"].tolist())
    b = set(frame.sort_values(col_b, ascending=False).head(k)["profile_id"].tolist())
    inter = len(a & b)
    union = len(a | b)
    return {
        "k": k,
        "intersection": inter,
        "jaccard": (inter / union) if union else 0.0,
    }


def write_markdown_report(
    output_path: Path,
    report: dict[str, object],
    top_profiles_prestige: pd.DataFrame,
    top_gap_positive: pd.DataFrame,
    top_gap_negative: pd.DataFrame,
) -> None:
    ds = report["dataset_summary"]
    full_corr = report["full_layer"]["correlations"]
    full_ineq = report["full_layer"]["profile_inequality"]
    pos_corr = report["positive_layer"]["correlations"]
    overlap = report["full_layer"]["top100_popularity_vs_prestige_overlap"]

    prestige_ids = ", ".join(str(x) for x in top_profiles_prestige["profile_id"].head(5).tolist())
    positive_gap_ids = ", ".join(str(x) for x in top_gap_positive["profile_id"].head(5).tolist())
    negative_gap_ids = ", ".join(str(x) for x in top_gap_negative["profile_id"].head(5).tolist())

    text = f"""# Popularity vs Prestige in an Online Dating Rating Network

## Research Question

Do the most **popular** profiles in the `rec-dating` platform also emerge as the most **prestigious** profiles once we account for who is rating them?

## Data and Modeling Choice

The dataset contains `{ds['edge_count']:,}` weighted interactions.  
We model it as a **role-based bipartite network**:

- `rater` nodes give ratings
- `profile` nodes receive ratings
- edge weight = rating from 1 to 10

This decision is important because the same raw numeric ID can play two different structural roles.

## Core Measures

- Popularity of profiles: `in_degree`, `in_strength`
- Prestige of profiles: HITS `authority_score`
- Activity of raters: `out_degree`, `out_strength`
- Structural importance of raters: HITS `hub_score`

## Main Findings

1. Popularity and prestige are strongly related, but not identical.
   Full-layer Pearson correlation between `in_strength` and `authority_score`: `{full_corr['pearson']:.4f}`.
   Full-layer Spearman correlation: `{full_corr['spearman']:.4f}`.

2. The top-100 popularity and top-100 prestige sets overlap, but not perfectly.
   Intersection: `{overlap['intersection']}`.
   Jaccard similarity: `{overlap['jaccard']:.4f}`.

3. Attention is highly unequal.
   Gini for profile in-strength: `{full_ineq['gini_in_strength']:.4f}`.
   Top 1% share of total in-strength: `{full_ineq['top_1pct_in_strength_share']:.4f}`.

4. Prestige is even more concentrated than raw popularity.
   Gini for authority: `{full_ineq['gini_authority']:.4f}`.
   Top 1% share of authority mass: `{full_ineq['top_1pct_authority_share']:.4f}`.

5. When we restrict the graph to strong positive ratings (`rating >= {report['positive_layer']['threshold']}`), popularity and prestige remain closely connected.
   Positive-layer Pearson correlation: `{pos_corr['pearson']:.4f}`.
   Positive-layer Spearman correlation: `{pos_corr['spearman']:.4f}`.

## Interpretation

The network does not behave like a simple raw-count popularity contest.  
Profiles that receive attention from structurally strong raters gain additional prestige through the bipartite network topology.  
This is why authority-based prestige and in-strength popularity overlap heavily but still diverge in meaningful cases.

## Concrete Cases

- Top prestige profiles include IDs: `{prestige_ids}`.
- Profiles with the largest positive prestige gap include IDs: `{positive_gap_ids}`.
- Profiles that are more popular than prestigious include IDs: `{negative_gap_ids}`.

## Limitations

- We do not have demographic or temporal metadata in the local file.
- IDs are anonymous, so the project focuses on structural patterns rather than user attributes.
- Because the platform semantics are limited, we interpret ratings as directed evaluative ties, not as confirmed matches or outcomes.

## Deliverables

- Tables and machine-readable outputs in `outputs/data/`
- Figures in `outputs/figures/`
- Reports in `outputs/reports/`
- Data preparation notebook in `notebooks/01_data_preparation.ipynb`
- Exploratory notebook in `notebooks/02_rec_dating_exploration.ipynb`
- Final summary notebook in `notebooks/03_final_project_analysis.ipynb`
"""
    output_path.write_text(text, encoding="utf-8")


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
    full_hits = full_analyzer.compute_hits(max_iter=200, tol=1e-7)
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

    report = {
        "dataset_summary": dataset_summary.to_dict(),
        "full_layer": {
            "hits_iterations": full_hits.iterations,
            "hits_converged": full_hits.converged,
            "snapshot": {
                "edge_count": full_snapshot.edge_count,
                "num_raters": full_snapshot.num_raters,
                "num_profiles": full_snapshot.num_profiles,
                "density": full_snapshot.density,
            },
            "correlations": full_analyzer.popularity_vs_prestige_correlation(profile_metrics),
            "profile_inequality": full_analyzer.profile_inequality_summary(profile_metrics),
            "rater_inequality": full_analyzer.rater_inequality_summary(rater_metrics),
            "top100_popularity_vs_prestige_overlap": top_overlap(
                profile_metrics, "in_strength", "authority_score", k=100
            ),
        },
        "positive_layer": {
            "threshold": args.positive_threshold,
            "snapshot": {
                "edge_count": positive_snapshot.edge_count,
                "num_raters": positive_snapshot.num_raters,
                "num_profiles": positive_snapshot.num_profiles,
                "density": positive_snapshot.density,
            },
            "correlations": positive_analyzer.popularity_vs_prestige_correlation(positive_profile_metrics),
            "profile_inequality": positive_analyzer.profile_inequality_summary(positive_profile_metrics),
            "top100_popularity_vs_prestige_overlap": top_overlap(
                positive_profile_metrics, "in_strength", "authority_score", k=100
            ),
        },
    }

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
    print(json.dumps(report["full_layer"]["correlations"], indent=2))


if __name__ == "__main__":
    main()
