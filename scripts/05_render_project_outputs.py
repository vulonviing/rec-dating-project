from __future__ import annotations

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

from rec_dating_project import ProjectPaths


LABEL = "full"


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
    y = cumulative / cumulative[-1] if cumulative[-1] != 0 else np.linspace(0, 1, cumulative.size)
    x = np.linspace(0, 1, cumulative.size)
    return x, y


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

Do the most popular profiles in the platform also become the most prestigious once we account for the structure of the bipartite rating network?

## Data

- Edges: `{ds['edge_count']:,}`
- Unique raters: `{ds['unique_raters']:,}`
- Unique profiles: `{ds['unique_profiles']:,}`
- Mean rating: `{ds['mean_rating']:.4f}`

## Modeling Choice

The dataset is analyzed as a **role-based bipartite graph**:

- `rater` role: gives scores
- `profile` role: receives scores

This separates behavioral activity from received attention and matches the logic of HITS:

- `hub_score` for raters
- `authority_score` for profiles

## Main Results

1. Popularity and prestige are strongly related in the full network.
   Pearson correlation: `{full_corr['pearson']:.4f}`.
   Spearman correlation: `{full_corr['spearman']:.4f}`.

2. The top-100 popularity and top-100 prestige sets overlap substantially but not perfectly.
   Intersection: `{overlap['intersection']}`.
   Jaccard similarity: `{overlap['jaccard']:.4f}`.

3. Attention is highly unequal.
   In-strength Gini: `{full_ineq['gini_in_strength']:.4f}`.
   Authority Gini: `{full_ineq['gini_authority']:.4f}`.
   Top 1% share of in-strength: `{full_ineq['top_1pct_in_strength_share']:.4f}`.
   Top 1% share of authority: `{full_ineq['top_1pct_authority_share']:.4f}`.

4. Restricting the graph to strong positive ratings preserves the same pattern.
   Positive-layer Pearson correlation: `{pos_corr['pearson']:.4f}`.
   Positive-layer Spearman correlation: `{pos_corr['spearman']:.4f}`.

## Interpretation

The most visible profiles are often also the most prestigious, but prestige is not reducible to raw counts alone.
Because authority depends on who is rating whom, structurally strong raters amplify some profiles more than others.

## Structural Cases

- Top prestige profile IDs: `{prestige_ids}`
- Highest positive prestige-gap IDs: `{positive_gap_ids}`
- Highest negative prestige-gap IDs: `{negative_gap_ids}`

## Limits

- No user metadata is available in the local dataset.
- No time stamps are available in the local edge file.
- Interpretation is structural rather than demographic or causal.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    paths = ProjectPaths.default()
    figures_dir = paths.output_figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((paths.output_reports_dir / f"final_project_report_{LABEL}.json").read_text())
    rating_dist = pd.read_csv(paths.output_data_dir / f"rating_distribution_{LABEL}.csv")
    profile_metrics = pd.read_csv(paths.output_data_dir / f"profile_metrics_{LABEL}.csv")
    rater_metrics = pd.read_csv(paths.output_data_dir / f"rater_metrics_{LABEL}.csv")
    top_profiles_prestige = pd.read_csv(paths.output_data_dir / f"top_profiles_prestige_{LABEL}.csv")
    top_gap_positive = pd.read_csv(paths.output_data_dir / f"top_profiles_prestige_gap_positive_{LABEL}.csv")
    top_gap_negative = pd.read_csv(paths.output_data_dir / f"top_profiles_prestige_gap_negative_{LABEL}.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(rating_dist["rating"], rating_dist["share"], color="#386641")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Share of Edges")
    plt.tight_layout()
    plt.savefig(figures_dir / f"rating_distribution_{LABEL}.png", dpi=220)
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
    plt.savefig(figures_dir / f"strength_ccdf_{LABEL}.png", dpi=220)
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
    plt.savefig(figures_dir / f"popularity_vs_prestige_{LABEL}.png", dpi=220)
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
    plt.savefig(figures_dir / f"profile_lorenz_{LABEL}.png", dpi=220)
    plt.close(fig)

    write_markdown_report(
        paths.output_reports_dir / f"project_report_{LABEL}.md",
        report=report,
        top_profiles_prestige=top_profiles_prestige,
        top_gap_positive=top_gap_positive,
        top_gap_negative=top_gap_negative,
    )

    print(f"Rendered figures and Markdown report in {paths.outputs_dir}")


if __name__ == "__main__":
    main()
