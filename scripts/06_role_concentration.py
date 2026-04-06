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

from rec_dating_project import PopularityPrestigeAnalyzer, ProjectPaths, RecDatingDataset, RoleBasedBipartiteNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure interaction concentration across rater and profile roles."
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional number of rows to read.")
    return parser.parse_args()


def run_label(nrows: int | None) -> str:
    return "full" if nrows is None else str(nrows)


def assign_percentile_bucket_labels(frame: pd.DataFrame, id_col: str, degree_col: str) -> pd.DataFrame:
    ranked = frame.sort_values(degree_col, ascending=False).reset_index(drop=True)[[id_col, degree_col]].copy()
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


def compute_top_rater_target_distribution(
    dataset: RecDatingDataset,
    profile_bucket_lookup: np.ndarray,
    rater_rank_lookup: np.ndarray,
    nrows: int | None = None,
) -> pd.DataFrame:
    group_specs = [
        ("All raters", 1.0),
        ("Top 20% raters", 0.20),
        ("Top 10% raters", 0.10),
        ("Top 5% raters", 0.05),
        ("Top 1% raters", 0.01),
    ]
    bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
    bucket_to_index = {label: idx for idx, label in enumerate(bucket_order)}

    edge_counts = {group: np.zeros(len(bucket_order), dtype=np.int64) for group, _ in group_specs}
    strength_sums = {group: np.zeros(len(bucket_order), dtype=np.float64) for group, _ in group_specs}
    total_edges = {group: 0 for group, _ in group_specs}
    total_strength = {group: 0.0 for group, _ in group_specs}

    for chunk in dataset.iter_chunks(nrows=nrows):
        rater_ids = chunk["rater_id"].to_numpy(dtype=np.int32, copy=False)
        profile_ids = chunk["profile_id"].to_numpy(dtype=np.int32, copy=False)
        ratings = chunk["rating"].to_numpy(dtype=np.float64, copy=False)

        chunk_ranks = rater_rank_lookup[rater_ids]
        chunk_profile_codes = profile_bucket_lookup[profile_ids]
        valid_profile_mask = chunk_profile_codes >= 0
        if not np.any(valid_profile_mask):
            continue

        chunk_ranks = chunk_ranks[valid_profile_mask]
        chunk_profile_codes = chunk_profile_codes[valid_profile_mask]
        ratings = ratings[valid_profile_mask]

        for group_name, threshold in group_specs:
            if threshold >= 1.0:
                mask = np.ones(chunk_ranks.shape[0], dtype=bool)
            else:
                mask = chunk_ranks <= threshold
            if not np.any(mask):
                continue
            counts = np.bincount(chunk_profile_codes[mask], minlength=len(bucket_order))
            sums = np.bincount(chunk_profile_codes[mask], weights=ratings[mask], minlength=len(bucket_order))
            edge_counts[group_name] += counts.astype(np.int64, copy=False)
            strength_sums[group_name] += sums.astype(np.float64, copy=False)
            total_edges[group_name] += int(mask.sum())
            total_strength[group_name] += float(ratings[mask].sum())

    all_edge_share = edge_counts["All raters"] / total_edges["All raters"] if total_edges["All raters"] else np.zeros(len(bucket_order))

    rows: list[dict[str, float | int | str]] = []
    for group_name, _ in group_specs:
        for bucket_label, bucket_idx in bucket_to_index.items():
            group_total_edges = total_edges[group_name]
            group_total_strength = total_strength[group_name]
            edge_share = (edge_counts[group_name][bucket_idx] / group_total_edges) if group_total_edges else 0.0
            strength_share = (strength_sums[group_name][bucket_idx] / group_total_strength) if group_total_strength else 0.0
            baseline_edge_share = float(all_edge_share[bucket_idx])
            rows.append(
                {
                    "rater_group": group_name,
                    "profile_bucket": bucket_label,
                    "edge_count": int(edge_counts[group_name][bucket_idx]),
                    "edge_share_within_group": float(edge_share),
                    "strength_share_within_group": float(strength_share),
                    "baseline_edge_share_all_raters": baseline_edge_share,
                    "edge_share_lift_vs_all": float(edge_share / baseline_edge_share) if baseline_edge_share else 0.0,
                }
            )

    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    dataset_summary: dict[str, object],
    summary_frame: pd.DataFrame,
    bucket_frame: pd.DataFrame,
    targeting_frame: pd.DataFrame,
) -> None:
    profile = summary_frame.loc[summary_frame["role"] == "profile"].iloc[0]
    rater = summary_frame.loc[summary_frame["role"] == "rater"].iloc[0]
    profile_bottom = bucket_frame[
        (bucket_frame["role"] == "profile") & (bucket_frame["bucket"] == "Bottom 50%")
    ].iloc[0]
    rater_bottom = bucket_frame[
        (bucket_frame["role"] == "rater") & (bucket_frame["bucket"] == "Bottom 50%")
    ].iloc[0]
    top20_raters_top20_profiles = targeting_frame[
        (targeting_frame["rater_group"] == "Top 20% raters")
        & (targeting_frame["profile_bucket"].isin(["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%"]))
    ]["edge_share_within_group"].sum()
    all_raters_top20_profiles = targeting_frame[
        (targeting_frame["rater_group"] == "All raters")
        & (targeting_frame["profile_bucket"].isin(["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%"]))
    ]["edge_share_within_group"].sum()
    top1_raters_top20_profiles = targeting_frame[
        (targeting_frame["rater_group"] == "Top 1% raters")
        & (targeting_frame["profile_bucket"].isin(["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%"]))
    ]["edge_share_within_group"].sum()
    top1_raters_bottom50_profiles = targeting_frame[
        (targeting_frame["rater_group"] == "Top 1% raters")
        & (targeting_frame["profile_bucket"] == "Bottom 50%")
    ]["edge_share_within_group"].sum()
    all_raters_bottom50_profiles = targeting_frame[
        (targeting_frame["rater_group"] == "All raters")
        & (targeting_frame["profile_bucket"] == "Bottom 50%")
    ]["edge_share_within_group"].sum()

    text = f"""# Role-Based Concentration Report

## Research Question

In the role-based bipartite version of `rec-dating`, is attention in the `profile` role concentrated in a small minority of users?

## Data

- Edges: `{dataset_summary['edge_count']:,}`
- Unique raters: `{dataset_summary['unique_raters']:,}`
- Unique profiles: `{dataset_summary['unique_profiles']:,}`

## Model

We keep the original project framing:

- `rater` nodes give ratings
- `profile` nodes receive ratings

The analysis below asks how concentrated interaction counts are inside each role.

## Main Findings

1. Profile attention is strongly concentrated.
   The top `1%` of profiles receive `{profile['top_1pct_degree_share']:.2%}` of all ratings by count.
   The top `10%` receive `{profile['top_10pct_degree_share']:.2%}`.
   The top `20%` receive `{profile['top_20pct_degree_share']:.2%}`.

2. The lower half of profiles receives very little attention.
   The bottom `50%` of profiles account for only `{profile_bottom['degree_share']:.2%}` of all received ratings.
   `{profile['share_degree_le_3']:.2%}` of profiles receive `3` or fewer ratings.

3. Rater activity is also unequal, but less extreme than profile attention.
   The top `20%` of raters produce `{rater['top_20pct_degree_share']:.2%}` of all outgoing ratings.
   The bottom `50%` of raters account for `{rater_bottom['degree_share']:.2%}` of all outgoing ratings.

4. A very small share of profiles is enough to explain most received attention.
   `{profile['user_share_for_50pct_degree']:.2%}` of profiles account for half of all received ratings.
   `{profile['user_share_for_80pct_degree']:.2%}` of profiles account for `80%` of all received ratings.

5. The most active raters do not simply pile onto the already most visible profiles.
   The top `20%` of raters send `{top20_raters_top20_profiles:.2%}` of their ratings to the top `20%` of profiles, versus `{all_raters_top20_profiles:.2%}` in the full baseline.
   The top `1%` of raters send `{top1_raters_top20_profiles:.2%}` to the top `20%` and `{top1_raters_bottom50_profiles:.2%}` to the bottom `50%`, versus only `{all_raters_bottom50_profiles:.2%}` for all raters.

## Interpretation

The answer to the core question is yes: in the `profile` role, a relatively small minority receives a disproportionately large share of ratings, while a large share of profiles receives comparatively little attention.
This conclusion follows directly from the role-based bipartite model and does not rely on unsupported identity matching across roles.
At the same time, the highest-activity raters are somewhat more exploratory than the aggregate baseline: they still mostly rate visible profiles, but they spread relatively more attention toward mid-tier and lower-visibility profiles than the full population does.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    paths = ProjectPaths.default()
    paths.ensure_output_dirs()
    label = run_label(args.nrows)

    dataset = RecDatingDataset(paths.raw_edges_path)
    network = RoleBasedBipartiteNetwork(dataset)
    dataset_summary = dataset.compute_summary(nrows=args.nrows)
    snapshot = network.build_sparse_rating_matrix(nrows=args.nrows, summary=dataset_summary)

    analyzer = PopularityPrestigeAnalyzer(snapshot)
    profile_metrics = analyzer.profile_metrics()
    rater_metrics = analyzer.rater_metrics()

    profile_summary = analyzer.role_concentration_summary(
        frame=profile_metrics,
        role="profile",
        degree_col="in_degree",
        strength_col="in_strength",
    )
    rater_summary = analyzer.role_concentration_summary(
        frame=rater_metrics,
        role="rater",
        degree_col="out_degree",
        strength_col="out_strength",
    )

    summary_frame = pd.DataFrame([profile_summary.to_dict(), rater_summary.to_dict()])
    profile_buckets = analyzer.role_percentile_buckets(
        frame=profile_metrics,
        role="profile",
        id_col="profile_id",
        degree_col="in_degree",
        strength_col="in_strength",
    )
    rater_buckets = analyzer.role_percentile_buckets(
        frame=rater_metrics,
        role="rater",
        id_col="rater_id",
        degree_col="out_degree",
        strength_col="out_strength",
    )
    bucket_frame = pd.concat([profile_buckets, rater_buckets], ignore_index=True)

    profile_bucket_labels = assign_percentile_bucket_labels(
        frame=profile_metrics,
        id_col="profile_id",
        degree_col="in_degree",
    )
    rater_rank_labels = assign_percentile_bucket_labels(
        frame=rater_metrics,
        id_col="rater_id",
        degree_col="out_degree",
    )

    profile_bucket_lookup = np.full(int(profile_metrics["profile_id"].max()) + 1, -1, dtype=np.int16)
    bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
    bucket_to_code = {label: idx for idx, label in enumerate(bucket_order)}
    profile_bucket_lookup[
        profile_bucket_labels["profile_id"].to_numpy(dtype=np.int32, copy=False)
    ] = profile_bucket_labels["bucket"].map(bucket_to_code).to_numpy(dtype=np.int16, copy=False)

    rater_rank_lookup = np.full(int(rater_metrics["rater_id"].max()) + 1, np.inf, dtype=np.float64)
    rater_rank_lookup[
        rater_rank_labels["rater_id"].to_numpy(dtype=np.int32, copy=False)
    ] = rater_rank_labels["rank_pct"].to_numpy(dtype=np.float64, copy=False)

    targeting_frame = compute_top_rater_target_distribution(
        dataset=dataset,
        profile_bucket_lookup=profile_bucket_lookup,
        rater_rank_lookup=rater_rank_lookup,
        nrows=args.nrows,
    )

    top_k = 25
    top_profiles = profile_metrics.sort_values("in_degree", ascending=False).head(top_k).copy()
    top_profiles["interaction_share"] = top_profiles["in_degree"] / profile_metrics["in_degree"].sum()
    top_profiles["rank"] = np.arange(1, len(top_profiles) + 1)

    top_raters = rater_metrics.sort_values("out_degree", ascending=False).head(top_k).copy()
    top_raters["interaction_share"] = top_raters["out_degree"] / rater_metrics["out_degree"].sum()
    top_raters["rank"] = np.arange(1, len(top_raters) + 1)

    summary_csv = paths.output_data_dir / f"role_concentration_summary_{label}.csv"
    buckets_csv = paths.output_data_dir / f"role_concentration_buckets_{label}.csv"
    targeting_csv = paths.output_data_dir / f"top_rater_profile_targeting_{label}.csv"
    top_profiles_csv = paths.output_data_dir / f"top_profiles_by_role_concentration_{label}.csv"
    top_raters_csv = paths.output_data_dir / f"top_raters_by_role_concentration_{label}.csv"
    report_json = paths.output_reports_dir / f"role_concentration_report_{label}.json"
    report_md = paths.output_reports_dir / f"role_concentration_report_{label}.md"
    figure_path = paths.output_figures_dir / f"role_concentration_{label}.png"
    targeting_figure_path = paths.output_figures_dir / f"top_rater_profile_targeting_{label}.png"

    summary_frame.to_csv(summary_csv, index=False)
    bucket_frame.to_csv(buckets_csv, index=False)
    targeting_frame.to_csv(targeting_csv, index=False)
    top_profiles.to_csv(top_profiles_csv, index=False)
    top_raters.to_csv(top_raters_csv, index=False)
    report_json.write_text(
        json.dumps(
            {
                "dataset_summary": dataset_summary.to_dict(),
                "role_summaries": summary_frame.to_dict(orient="records"),
                "bucket_table": bucket_frame.to_dict(orient="records"),
                "targeting_table": targeting_frame.to_dict(orient="records"),
                "top_profiles": top_profiles.to_dict(orient="records"),
                "top_raters": top_raters.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(
        output_path=report_md,
        dataset_summary=dataset_summary.to_dict(),
        summary_frame=summary_frame,
        bucket_frame=bucket_frame,
        targeting_frame=targeting_frame,
    )

    top_share_labels = ["top_1pct_degree_share", "top_5pct_degree_share", "top_10pct_degree_share", "top_20pct_degree_share"]
    top_share_display = ["Top 1%", "Top 5%", "Top 10%", "Top 20%"]
    profile_top = summary_frame.loc[summary_frame["role"] == "profile", top_share_labels].iloc[0].to_numpy(dtype=float)
    rater_top = summary_frame.loc[summary_frame["role"] == "rater", top_share_labels].iloc[0].to_numpy(dtype=float)

    bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
    profile_bucket = (
        bucket_frame[bucket_frame["role"] == "profile"]
        .set_index("bucket")
        .reindex(bucket_order)["degree_share"]
        .to_numpy(dtype=float)
    )
    rater_bucket = (
        bucket_frame[bucket_frame["role"] == "rater"]
        .set_index("bucket")
        .reindex(bucket_order)["degree_share"]
        .to_numpy(dtype=float)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(top_share_display))
    width = 0.35
    axes[0].bar(x - width / 2, profile_top, width=width, label="Profiles", color="#1d3557")
    axes[0].bar(x + width / 2, rater_top, width=width, label="Raters", color="#e76f51")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top_share_display)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of total interaction count")
    axes[0].set_title("Top Shares by Role")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    x_bucket = np.arange(len(bucket_order))
    axes[1].bar(x_bucket - width / 2, profile_bucket, width=width, label="Profiles", color="#457b9d")
    axes[1].bar(x_bucket + width / 2, rater_bucket, width=width, label="Raters", color="#f4a261")
    axes[1].set_xticks(x_bucket)
    axes[1].set_xticklabels(bucket_order, rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Share of total interaction count")
    axes[1].set_title("Rank Buckets by Role")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close(fig)

    targeting_groups = ["All raters", "Top 20% raters", "Top 10% raters", "Top 5% raters", "Top 1% raters"]
    targeting_bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
    targeting_matrix = (
        targeting_frame.pivot(index="rater_group", columns="profile_bucket", values="edge_share_within_group")
        .reindex(index=targeting_groups, columns=targeting_bucket_order)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottom = np.zeros(len(targeting_groups))
    colors = ["#1d3557", "#457b9d", "#a8dadc", "#f4a261", "#e76f51", "#b56576"]
    for bucket_label, color in zip(targeting_bucket_order, colors):
        values = targeting_matrix[bucket_label].to_numpy(dtype=float)
        ax.bar(targeting_groups, values, bottom=bottom, label=bucket_label, color=color)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of outgoing ratings")
    ax.set_title("Where Top Raters Send Their Ratings")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Profile bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(targeting_figure_path, dpi=220)
    plt.close(fig)

    print(f"Saved role concentration outputs to {paths.outputs_dir}")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
