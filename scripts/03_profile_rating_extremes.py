from __future__ import annotations

import argparse
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

from rec_dating_project import PopularityPrestigeAnalyzer, ProjectPaths, RecDatingDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether a small subset of profiles concentrates high, low, and total received ratings."
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional number of rows to read.")
    parser.add_argument(
        "--high-min",
        type=int,
        default=8,
        help="Minimum rating treated as high.",
    )
    parser.add_argument(
        "--low-max",
        type=int,
        default=3,
        help="Maximum rating treated as low.",
    )
    return parser.parse_args()


def run_label(nrows: int | None) -> str:
    return "full" if nrows is None else str(nrows)


def assign_percentile_bucket_labels(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ranked = frame.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    ranked["rank_pct"] = (np.arange(len(ranked)) + 1) / len(ranked)

    bucket_specs = [
        (0.0, 0.01, "Top 1%"),
        (0.01, 0.05, "Top 1-5%"),
        (0.05, 0.10, "Top 5-10%"),
        (0.10, 0.20, "Top 10-20%"),
        (0.20, 0.50, "Top 20-50%"),
        (0.50, 1.00, "Bottom 50%"),
    ]
    ranked["bucket"] = "Unassigned"
    for start, end, label in bucket_specs:
        mask = (ranked["rank_pct"] > start) & (ranked["rank_pct"] <= end)
        ranked.loc[mask, "bucket"] = label
    return ranked


def value_bucket_table(frame: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
    ranked = assign_percentile_bucket_labels(frame, value_col=value_col)
    bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
    total_value = float(ranked[value_col].sum())
    rows: list[dict[str, float | int | str]] = []
    for bucket in bucket_order:
        subset = ranked[ranked["bucket"] == bucket]
        rows.append(
            {
                "series": label,
                "bucket": bucket,
                "profile_count": int(len(subset)),
                "profile_share": float(len(subset) / len(ranked)) if len(ranked) else 0.0,
                "value_share": float(subset[value_col].sum() / total_value) if total_value else 0.0,
                "mean_value": float(subset[value_col].mean()) if not subset.empty else 0.0,
                "median_value": float(subset[value_col].median()) if not subset.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def concentration_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array >= 0)]
    if array.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    sorted_desc = np.sort(array)[::-1]
    cumulative = np.cumsum(sorted_desc)
    total = cumulative[-1]
    x = np.arange(1, len(sorted_desc) + 1, dtype=np.float64) / len(sorted_desc)
    y = cumulative / total if total else np.zeros_like(x)
    return x, y


def top_share_summary(frame: pd.DataFrame, value_col: str, label: str) -> dict[str, float | int | str]:
    values = frame[value_col].to_numpy(dtype=np.float64, copy=False)
    any_positive = values > 0
    return {
        "series": label,
        "active_profiles": int(len(frame)),
        "profiles_with_any": int(any_positive.sum()),
        "profiles_with_any_share": float(any_positive.mean()) if len(frame) else 0.0,
        "total_events": int(values.sum()),
        "gini": PopularityPrestigeAnalyzer.gini(values),
        "top_1pct_share": PopularityPrestigeAnalyzer.top_share(values, fraction=0.01),
        "top_5pct_share": PopularityPrestigeAnalyzer.top_share(values, fraction=0.05),
        "top_10pct_share": PopularityPrestigeAnalyzer.top_share(values, fraction=0.10),
        "top_20pct_share": PopularityPrestigeAnalyzer.top_share(values, fraction=0.20),
        "user_share_for_50pct": PopularityPrestigeAnalyzer.fraction_needed_for_share(values, target_share=0.5),
        "user_share_for_80pct": PopularityPrestigeAnalyzer.fraction_needed_for_share(values, target_share=0.8),
        "median_value": float(np.median(values)) if len(values) else 0.0,
        "mean_value": float(np.mean(values)) if len(values) else 0.0,
        "p80_value": float(np.quantile(values, 0.8)) if len(values) else 0.0,
        "p95_value": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "max_value": int(values.max()) if len(values) else 0,
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.default()
    paths.ensure_output_dirs()
    label = run_label(args.nrows)

    dataset = RecDatingDataset(paths.raw_edges_path)
    summary = dataset.compute_summary(nrows=args.nrows)
    size = summary.max_profile_id + 1

    total_received = np.zeros(size, dtype=np.int64)
    high_received = np.zeros(size, dtype=np.int64)
    low_received = np.zeros(size, dtype=np.int64)
    total_rating_sum = np.zeros(size, dtype=np.float64)

    for chunk in dataset.iter_chunks(nrows=args.nrows):
        profile_ids = chunk["profile_id"].to_numpy(dtype=np.int32, copy=False)
        ratings = chunk["rating"].to_numpy(dtype=np.int16, copy=False)

        total_received += np.bincount(profile_ids, minlength=size)
        total_rating_sum += np.bincount(profile_ids, weights=ratings, minlength=size)

        high_mask = ratings >= args.high_min
        if np.any(high_mask):
            high_received += np.bincount(profile_ids[high_mask], minlength=size)

        low_mask = ratings <= args.low_max
        if np.any(low_mask):
            low_received += np.bincount(profile_ids[low_mask], minlength=size)

    active_mask = total_received > 0
    profile_frame = pd.DataFrame(
        {
            "profile_id": np.arange(size, dtype=np.int32),
            "total_received": total_received,
            "high_received": high_received,
            "low_received": low_received,
            "mean_rating": np.divide(
                total_rating_sum,
                total_received,
                out=np.zeros_like(total_rating_sum, dtype=np.float64),
                where=total_received > 0,
            ),
        }
    )
    profile_frame = profile_frame[active_mask].copy()
    profile_frame["high_share_within_profile"] = np.divide(
        profile_frame["high_received"],
        profile_frame["total_received"],
        out=np.zeros(len(profile_frame), dtype=np.float64),
        where=profile_frame["total_received"] > 0,
    )
    profile_frame["low_share_within_profile"] = np.divide(
        profile_frame["low_received"],
        profile_frame["total_received"],
        out=np.zeros(len(profile_frame), dtype=np.float64),
        where=profile_frame["total_received"] > 0,
    )

    summary_frame = pd.DataFrame(
        [
            top_share_summary(profile_frame, "total_received", "all_interactions"),
            top_share_summary(profile_frame, "high_received", "high_ratings"),
            top_share_summary(profile_frame, "low_received", "low_ratings"),
        ]
    )

    bucket_frame = pd.concat(
        [
            value_bucket_table(profile_frame, "total_received", "all_interactions"),
            value_bucket_table(profile_frame, "high_received", "high_ratings"),
            value_bucket_table(profile_frame, "low_received", "low_ratings"),
        ],
        ignore_index=True,
    )

    top_high = profile_frame.sort_values(["high_received", "high_share_within_profile"], ascending=False).head(25).copy()
    top_high["rank"] = np.arange(1, len(top_high) + 1)
    top_low = profile_frame.sort_values(["low_received", "low_share_within_profile"], ascending=False).head(25).copy()
    top_low["rank"] = np.arange(1, len(top_low) + 1)

    summary_csv = paths.output_data_dir / f"profile_rating_extremes_summary_{label}.csv"
    buckets_csv = paths.output_data_dir / f"profile_rating_extremes_buckets_{label}.csv"
    profile_csv = paths.output_data_dir / f"profile_rating_extremes_profiles_{label}.csv"
    top_high_csv = paths.output_data_dir / f"top_profiles_high_ratings_{label}.csv"
    top_low_csv = paths.output_data_dir / f"top_profiles_low_ratings_{label}.csv"
    figure_high_low = paths.output_figures_dir / f"profile_high_low_rating_concentration_{label}.png"
    figure_rating_curves = paths.output_figures_dir / f"profile_rating_concentration_curves_{label}.png"
    figure_bucket_shares = paths.output_figures_dir / f"profile_bucket_shares_{label}.png"
    figure_interaction_curve = paths.output_figures_dir / f"profile_interaction_concentration_curve_{label}.png"
    figure_interaction_buckets = paths.output_figures_dir / f"profile_interaction_bucket_shares_{label}.png"

    profile_frame.to_csv(profile_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    bucket_frame.to_csv(buckets_csv, index=False)
    top_high.to_csv(top_high_csv, index=False)
    top_low.to_csv(top_low_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    top_profile_shares = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.00], dtype=np.float64)
    x_labels = ["1%", "5%", "10%", "20%", "50%", "100%"]

    series_specs = [
        ("all_interactions", "total_received", "#457b9d", "All received interactions"),
        ("high_ratings", "high_received", "#2a9d8f", f"High ratings ({args.high_min}-10)"),
        ("low_ratings", "low_received", "#e76f51", f"Low ratings (1-{args.low_max})"),
    ]

    for _, value_col, color, label_text in series_specs:
        y_values = [
            PopularityPrestigeAnalyzer.top_share(profile_frame[value_col], fraction=fraction)
            for fraction in top_profile_shares
        ]
        axes[0].plot(x_labels, y_values, marker="o", linewidth=2.2, color=color, label=label_text)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of received events")
    axes[0].set_xlabel("Top share of profiles")
    axes[0].set_title("Do Small Profile Groups Concentrate High or Low Ratings?")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    grouped_cols = ["top_1pct_share", "top_5pct_share", "top_10pct_share", "top_20pct_share"]
    grouped_x = np.arange(len(grouped_cols))
    width = 0.24
    for offset, (series_name, color, label_text) in enumerate(
        [
            ("all_interactions", "#457b9d", "All"),
            ("high_ratings", "#2a9d8f", "High"),
            ("low_ratings", "#e76f51", "Low"),
        ]
    ):
        values = (
            summary_frame.loc[summary_frame["series"] == series_name, grouped_cols]
            .iloc[0]
            .to_numpy(dtype=float)
        )
        axes[1].bar(grouped_x + (offset - 1) * width, values, width=width, color=color, label=label_text)
    axes[1].set_xticks(grouped_x)
    axes[1].set_xticklabels(["Top 1%", "Top 5%", "Top 10%", "Top 20%"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share of received events")
    axes[1].set_title("Top Profile Groups by Event Type")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(figure_high_low, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for value_col, color, label_text in [
        ("high_received", "#2a9d8f", f"High ratings ({args.high_min}-10)"),
        ("low_received", "#e76f51", f"Low ratings (1-{args.low_max})"),
    ]:
        x_curve, y_curve = concentration_curve(profile_frame[value_col].to_numpy(dtype=np.float64, copy=False))
        ax.plot(x_curve, y_curve, color=color, linewidth=2.2, label=label_text)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Top share of profiles")
    ax.set_ylabel("Cumulative share of received ratings")
    ax.set_title("High and Low Rating Concentration Across Profiles")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figure_rating_curves, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
    grouped_x = np.arange(len(bucket_order))
    width = 0.34
    for offset, (series_name, color, label_text) in enumerate(
        [
            ("high_ratings", "#2a9d8f", "High"),
            ("low_ratings", "#e76f51", "Low"),
        ]
    ):
        values = (
            bucket_frame[bucket_frame["series"] == series_name]
            .set_index("bucket")
            .reindex(bucket_order)["value_share"]
            .to_numpy(dtype=float)
        )
        ax.bar(grouped_x + (offset - 0.5) * width, values, width=width, color=color, label=label_text)
    ax.set_xticks(grouped_x)
    ax.set_xticklabels(bucket_order, rotation=20)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Share of received events")
    ax.set_title("How Much Does Each Profile Bucket Capture of High or Low Ratings?")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figure_bucket_shares, dpi=220)
    plt.close(fig)

    x_curve, y_curve = concentration_curve(profile_frame["total_received"].to_numpy(dtype=np.float64, copy=False))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x_curve, y_curve, color="#1d3557", linewidth=2.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Top share of profiles")
    ax.set_ylabel("Cumulative share of received interactions")
    ax.set_title("Overall Interaction Concentration Across Profiles")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_interaction_curve, dpi=220)
    plt.close(fig)

    overall_buckets = bucket_frame[bucket_frame["series"] == "all_interactions"].copy()
    overall_buckets = overall_buckets.set_index("bucket").reindex(bucket_order).reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(
        overall_buckets["bucket"],
        overall_buckets["value_share"],
        color=["#1d3557", "#457b9d", "#5c7ea7", "#7b9cc0", "#a8c0d9", "#d9e7f5"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of all received interactions")
    ax.set_title("Which Profile Buckets Capture Most Interactions?")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_interaction_buckets, dpi=220)
    plt.close(fig)

    print(f"Saved profile rating extremes outputs to {paths.outputs_dir}")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
