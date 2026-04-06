from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import hypergeom, t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_dating_project import (
    PopularityPrestigeAnalyzer,
    ProjectPaths,
    RecDatingDataset,
    RoleBasedBipartiteNetwork,
)


FEATURE_SPECS: list[tuple[str, str, str]] = [
    ("full_in_degree", "Full in-degree", "full"),
    ("full_in_strength", "Full in-strength", "full"),
    ("full_mean_rating", "Full mean rating", "full"),
    ("full_authority_score", "Full authority", "full"),
    ("full_popularity_rank_pct", "Full popularity rank", "full"),
    ("full_prestige_rank_pct", "Full prestige rank", "full"),
    ("full_prestige_gap", "Full prestige gap", "full"),
    ("pos8_in_degree", "Pos8 in-degree", "pos8"),
    ("pos8_in_strength", "Pos8 in-strength", "pos8"),
    ("pos8_mean_rating", "Pos8 mean rating", "pos8"),
    ("pos8_authority_score", "Pos8 authority", "pos8"),
    ("pos8_popularity_rank_pct", "Pos8 popularity rank", "pos8"),
    ("pos8_prestige_rank_pct", "Pos8 prestige rank", "pos8"),
    ("pos8_prestige_gap", "Pos8 prestige gap", "pos8"),
    ("pos3_in_degree", "Pos3 in-degree", "pos3"),
    ("pos3_in_strength", "Pos3 in-strength", "pos3"),
    ("pos3_mean_rating", "Pos3 mean rating", "pos3"),
    ("pos3_authority_score", "Pos3 authority", "pos3"),
    ("pos3_popularity_rank_pct", "Pos3 popularity rank", "pos3"),
    ("pos3_prestige_rank_pct", "Pos3 prestige rank", "pos3"),
    ("pos3_prestige_gap", "Pos3 prestige gap", "pos3"),
]


BUCKET_SPECS: list[tuple[float, float, str]] = [
    (0.0, 0.01, "Top 1%"),
    (0.01, 0.05, "Top 1-5%"),
    (0.05, 0.10, "Top 5-10%"),
    (0.10, 0.20, "Top 10-20%"),
    (0.20, 0.50, "Top 20-50%"),
    (0.50, 1.00, "Bottom 50%"),
]


BUCKET_ANALYSIS_SPECS: list[tuple[str, str]] = [
    ("Top 1%", "top_1"),
    ("Top 1-5%", "top_1_5"),
    ("Top 5-10%", "top_5_10"),
    ("Top 10-20%", "top_10_20"),
    ("Top 20-50%", "top_20_50"),
    ("Bottom 50%", "bottom_50"),
]


GROUP_SPECS: list[tuple[str, str, str, str]] = [
    (f"interaction_{key}", f"Interaction {label}", "interaction_bucket", label)
    for label, key in BUCKET_ANALYSIS_SPECS
] + [
    (f"high_{key}", f"High {label}", "high_bucket", label)
    for label, key in BUCKET_ANALYSIS_SPECS
]

FEATURE_CLASS_SPECS: list[tuple[str, dict[str, list[str]]]] = [
    (
        "Volume / Exposure",
        {
            "full": ["full_in_degree", "full_in_strength"],
            "pos8": ["pos8_in_degree", "pos8_in_strength"],
            "pos3": ["pos3_in_degree", "pos3_in_strength"],
        },
    ),
    (
        "Prestige / Centrality",
        {
            "full": ["full_authority_score"],
            "pos8": ["pos8_authority_score"],
            "pos3": ["pos3_authority_score"],
        },
    ),
    (
        "Rank Signals",
        {
            "full": ["full_popularity_rank_pct", "full_prestige_rank_pct"],
            "pos8": ["pos8_popularity_rank_pct", "pos8_prestige_rank_pct"],
            "pos3": ["pos3_popularity_rank_pct", "pos3_prestige_rank_pct"],
        },
    ),
    (
        "Rating Level",
        {
            "full": ["full_mean_rating"],
            "pos8": ["pos8_mean_rating"],
            "pos3": ["pos3_mean_rating"],
        },
    ),
    (
        "Gap / Misalignment",
        {
            "full": ["full_prestige_gap"],
            "pos8": ["pos8_prestige_gap"],
            "pos3": ["pos3_prestige_gap"],
        },
    ),
]

NULL_MODEL_DRAWS = 20_000
NULL_MODEL_SEED = 20260406
LOW_MAX_RATING = 3


def assign_percentile_buckets(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    ranked = frame[["profile_id", value_col]].sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    ranked["rank_pct"] = (np.arange(len(ranked)) + 1) / len(ranked)
    ranked["bucket"] = "Unassigned"
    for start, end, label in BUCKET_SPECS:
        mask = (ranked["rank_pct"] > start) & (ranked["rank_pct"] <= end)
        ranked.loc[mask, "bucket"] = label
    return ranked[["profile_id", "bucket"]]


def point_biserial_stats(values: pd.Series, membership: pd.Series) -> tuple[float, float, int]:
    valid = values.notna() & membership.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, 1.0, 0

    binary = membership[valid].astype(int)
    if binary.nunique() < 2:
        return 0.0, 1.0, n_valid

    corr = values[valid].corr(binary, method="pearson")
    corr = float(corr) if pd.notna(corr) else 0.0
    corr = float(np.clip(corr, -0.999999999999, 0.999999999999))
    if n_valid <= 2:
        return corr, 1.0, n_valid

    t_stat = corr * np.sqrt((n_valid - 2) / max(1e-12, 1.0 - corr * corr))
    p_value = float(2.0 * t.sf(abs(t_stat), df=n_valid - 2))
    return corr, p_value, n_valid


def overlap_null_stats(
    population_size: int,
    size_a: int,
    size_b: int,
    observed_intersection: int,
    rng: np.random.Generator,
    draws: int = NULL_MODEL_DRAWS,
) -> dict[str, float | int]:
    distribution = hypergeom(M=population_size, n=size_a, N=size_b)
    expected_intersection = float(distribution.mean())
    expected_std = float(distribution.std())
    exact_p_value = float(distribution.sf(observed_intersection - 1))
    log_p_value = float(distribution.logsf(observed_intersection - 1))
    simulated = rng.hypergeometric(size_a, population_size - size_a, size_b, size=draws)
    simulated_mean = float(simulated.mean())
    simulated_std = float(simulated.std(ddof=0))
    simulated_p_ge_observed = float((simulated >= observed_intersection).mean())

    return {
        "population_size": int(population_size),
        "expected_random_intersection": expected_intersection,
        "expected_random_overlap_share_interaction": float(expected_intersection / size_a) if size_a else 0.0,
        "expected_random_overlap_share_high": float(expected_intersection / size_b) if size_b else 0.0,
        "expected_random_jaccard": float(expected_intersection / (size_a + size_b - expected_intersection))
        if (size_a + size_b - expected_intersection) > 0
        else 0.0,
        "exact_hypergeom_p_value": exact_p_value,
        "exact_hypergeom_log10_p_value": float(log_p_value / np.log(10)) if np.isfinite(log_p_value) else float("-inf"),
        "simulation_draws": int(draws),
        "simulated_mean_intersection": simulated_mean,
        "simulated_std_intersection": simulated_std,
        "simulated_max_intersection": int(simulated.max()) if simulated.size else 0,
        "simulated_p_ge_observed": simulated_p_ge_observed,
        "zscore_vs_random": float((observed_intersection - expected_intersection) / expected_std) if expected_std > 0 else 0.0,
    }


def build_pos3_profile_metrics(paths: ProjectPaths) -> pd.DataFrame:
    dataset = RecDatingDataset(paths.raw_edges_path)
    network = RoleBasedBipartiteNetwork(dataset)
    snapshot = network.build_sparse_rating_matrix(
        min_rating=1,
        max_rating=LOW_MAX_RATING,
    )
    analyzer = PopularityPrestigeAnalyzer(snapshot)
    return analyzer.profile_metrics()


def zscore_series(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
    return (values - values.mean()) / std


def plot_heatmap(
    pivot: pd.DataFrame,
    output_path: Path,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
) -> None:
    fig_height = max(5.0, 0.45 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                format(matrix[row, col], fmt),
                ha="center",
                va="center",
                fontsize=8.5,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def build_bucket_pair_frame(correlation_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for feature_col, feature_label, family in FEATURE_SPECS:
        subset = correlation_frame[correlation_frame["feature"] == feature_col]
        for bucket_label, _ in BUCKET_ANALYSIS_SPECS:
            interaction_corr = subset.loc[
                subset["group_label"] == f"Interaction {bucket_label}", "membership_corr"
            ].iloc[0]
            high_corr = subset.loc[subset["group_label"] == f"High {bucket_label}", "membership_corr"].iloc[0]
            rows.append(
                {
                    "feature": feature_col,
                    "feature_label": feature_label,
                    "family": family,
                    "bucket": bucket_label,
                    "interaction_corr": float(interaction_corr),
                    "high_corr": float(high_corr),
                }
            )
    return pd.DataFrame(rows)


def select_label_points(bucket_pair_frame: pd.DataFrame) -> pd.DataFrame:
    frame = bucket_pair_frame.copy()
    frame["sum_corr"] = frame["interaction_corr"] + frame["high_corr"]
    frame["diag_gap"] = (frame["high_corr"] - frame["interaction_corr"]).abs()
    frame["dist_origin"] = np.hypot(frame["interaction_corr"], frame["high_corr"])

    selected_parts: list[pd.DataFrame] = []
    for family in ("full", "pos8", "pos3"):
        family_frame = frame[frame["family"] == family].copy()
        positive = family_frame[family_frame["sum_corr"] > 0].nlargest(2, "dist_origin").copy()
        positive["label_reason"] = "strong_positive_alignment"

        negative = family_frame[family_frame["sum_corr"] < 0].nlargest(2, "dist_origin").copy()
        negative["label_reason"] = "strong_negative_alignment"

        off_diagonal = family_frame.nlargest(1, "diag_gap").copy()
        off_diagonal["label_reason"] = "off_diagonal_outlier"

        selected_parts.extend([positive, negative, off_diagonal])

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["dist_origin", "diag_gap"], ascending=[False, False])
    selected = selected.drop_duplicates(subset=["feature", "family", "bucket"]).reset_index(drop=True)
    return selected


def write_bucket_pair_markdown(bucket_pair_frame: pd.DataFrame, selected_points: pd.DataFrame, output_path: Path) -> None:
    display = bucket_pair_frame.copy()
    selected_keys = {
        (row.feature, row.family, row.bucket): row.label_reason
        for row in selected_points.itertuples(index=False)
    }
    display["selected_label"] = [
        "yes" if (feature, family, bucket) in selected_keys else "no"
        for feature, family, bucket in zip(display["feature"], display["family"], display["bucket"])
    ]
    display["label_reason"] = [
        selected_keys.get((feature, family, bucket), "")
        for feature, family, bucket in zip(display["feature"], display["family"], display["bucket"])
    ]
    display["interaction_corr"] = display["interaction_corr"].map(lambda x: f"{x:.3f}")
    display["high_corr"] = display["high_corr"].map(lambda x: f"{x:.3f}")
    display["family"] = display["family"].str.upper()

    header = [
        "# Feature Alignment Point Table",
        "",
        "All plotted points from the consistency scatter. `selected_label = yes` marks the points annotated in the figure.",
        "",
        "| Feature | Family | Bucket | Interaction Corr | High Corr | Selected Label | Label Reason |",
        "|---|---|---|---:|---:|---|---|",
    ]
    rows = [
        f"| {row.feature_label} | {row.family} | {row.bucket} | {row.interaction_corr} | {row.high_corr} | {row.selected_label} | {row.label_reason or '-'} |"
        for row in display.itertuples(index=False)
    ]
    output_path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def build_feature_class_frame(summary_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    layer_labels = {"full": "ALL", "pos8": "POS8", "pos3": "POS3"}
    for class_label, family_map in FEATURE_CLASS_SPECS:
        for family, feature_names in family_map.items():
            subset = summary_frame[
                (summary_frame["family"] == family) & (summary_frame["feature"].isin(feature_names))
            ].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    "class_label": class_label,
                    "layer": family,
                    "layer_label": layer_labels[family],
                    "feature_count": int(len(subset)),
                    "interaction_mean_corr": float(subset["interaction_mean_corr"].mean()),
                    "high_mean_corr": float(subset["high_mean_corr"].mean()),
                    "interaction_min_corr": float(subset["interaction_mean_corr"].min()),
                    "interaction_max_corr": float(subset["interaction_mean_corr"].max()),
                    "high_min_corr": float(subset["high_mean_corr"].min()),
                    "high_max_corr": float(subset["high_mean_corr"].max()),
                    "interaction_std_corr": float(subset["interaction_mean_corr"].std(ddof=0)),
                    "high_std_corr": float(subset["high_mean_corr"].std(ddof=0)),
                    "included_features": "; ".join(subset["feature_label"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def write_feature_class_markdown(feature_class_frame: pd.DataFrame, output_path: Path) -> None:
    display = feature_class_frame.copy()
    for col in [
        "interaction_mean_corr",
        "high_mean_corr",
        "interaction_min_corr",
        "interaction_max_corr",
        "high_min_corr",
        "high_max_corr",
    ]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")

    header = [
        "# Feature Class Summary",
        "",
        "Each row aggregates several original variables into one interpretable feature class within one layer.",
        "",
        "| Feature Class | Layer | N Features | Mean Interaction Corr | Mean High Corr | Min Interaction | Max Interaction | Min High | Max High | Included Features |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = [
        f"| {row.class_label} | {row.layer_label} | {row.feature_count} | {row.interaction_mean_corr} | {row.high_mean_corr} | {row.interaction_min_corr} | {row.interaction_max_corr} | {row.high_min_corr} | {row.high_max_corr} | {row.included_features} |"
        for row in display.itertuples(index=False)
    ]
    output_path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def plot_feature_class_scatter(feature_class_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    class_colors = {
        "Volume / Exposure": "#c0392b",
        "Prestige / Centrality": "#2a9d8f",
        "Rank Signals": "#f4a261",
        "Rating Level": "#577590",
        "Gap / Misalignment": "#7f8c8d",
    }
    layer_markers = {"ALL": "o", "POS8": "s", "POS3": "^"}

    for row in feature_class_frame.itertuples(index=False):
        xerr = np.array(
            [
                [row.interaction_mean_corr - row.interaction_min_corr],
                [row.interaction_max_corr - row.interaction_mean_corr],
            ]
        )
        yerr = np.array(
            [
                [row.high_mean_corr - row.high_min_corr],
                [row.high_max_corr - row.high_mean_corr],
            ]
        )
        color = class_colors.get(row.class_label, "#555555")
        marker = layer_markers[row.layer_label]
        ax.errorbar(
            row.interaction_mean_corr,
            row.high_mean_corr,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            markersize=9,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            alpha=0.92,
        )

    max_abs = float(
        max(
            feature_class_frame["interaction_max_corr"].abs().max(),
            feature_class_frame["high_max_corr"].abs().max(),
            feature_class_frame["interaction_min_corr"].abs().max(),
            feature_class_frame["high_min_corr"].abs().max(),
        )
    )
    lim = max_abs * 1.08

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Mean correlation with interaction bucket membership")
    ax.set_ylabel("Mean correlation with high-rating bucket membership")
    ax.set_title("How Do Feature Classes Align Across the Bucket Ladder?")
    ax.grid(alpha=0.28)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_label,
            markerfacecolor=class_colors[class_label],
            markeredgecolor=class_colors[class_label],
            markersize=8,
        )
        for class_label, _ in FEATURE_CLASS_SPECS
    ]
    class_legend = ax.legend(handles=class_handles, title="Feature Class", loc="upper left")
    ax.add_artist(class_legend)
    layer_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="0.25",
            label=layer_label,
            markerfacecolor="white",
            markeredgecolor="0.25",
            linestyle="None",
            markersize=8,
        )
        for layer_label, marker in layer_markers.items()
    ]
    ax.legend(handles=layer_handles, title="Layer", loc="upper left", bbox_to_anchor=(0.0, 0.62))
    ax.text(
        0.99,
        0.02,
        "Each point is a class mean within one layer.\nError bars show min-max spread of included variables.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.8"},
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_consistency_scatter(
    bucket_pair_frame: pd.DataFrame,
    selected_points: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.4))
    bucket_colors = {
        "Top 1%": "#b22222",
        "Top 1-5%": "#e76f51",
        "Top 5-10%": "#f4a261",
        "Top 10-20%": "#e9c46a",
        "Top 20-50%": "#5c7c8a",
        "Bottom 50%": "#264653",
    }
    family_markers = {"full": "o", "pos8": "s", "pos3": "^"}
    label_offsets = [
        (8, 8),
        (8, -9),
        (-8, 8),
        (-8, -9),
        (10, 2),
        (-10, 2),
        (2, 10),
        (2, -10),
    ]
    selected_lookup = {
        (row.feature, row.family, row.bucket): idx
        for idx, row in enumerate(selected_points.itertuples(index=False))
    }

    for bucket_label, _ in BUCKET_ANALYSIS_SPECS:
        bucket_subset = bucket_pair_frame[bucket_pair_frame["bucket"] == bucket_label]
        for family, marker in family_markers.items():
            subset = bucket_subset[bucket_subset["family"] == family]
            if subset.empty:
                continue
            ax.scatter(
                subset["interaction_corr"],
                subset["high_corr"],
                s=72,
                alpha=0.88,
                color=bucket_colors[bucket_label],
                edgecolors="black",
                linewidths=0.35,
                marker=marker,
                label=bucket_label if family == "full" else None,
            )
            for row in subset.itertuples(index=False):
                key = (row.feature, row.family, row.bucket)
                if key not in selected_lookup:
                    continue
                dx, dy = label_offsets[selected_lookup[key] % len(label_offsets)]
                ax.annotate(
                    row.feature_label,
                    (row.interaction_corr, row.high_corr),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.1,
                    color="black",
                    alpha=0.88,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "alpha": 0.72,
                        "edgecolor": "none",
                    },
                )

    max_abs = float(
        max(
            bucket_pair_frame["interaction_corr"].abs().max(),
            bucket_pair_frame["high_corr"].abs().max(),
        )
    )
    lim = max_abs * 1.08

    ax.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Correlation with interaction bucket membership")
    ax.set_ylabel("Correlation with high-rating bucket membership")
    ax.set_title("Do Matching Interaction and High-Rating Buckets Align in Feature Space?")
    ax.grid(alpha=0.28)

    bucket_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=bucket_label,
            markerfacecolor=bucket_colors[bucket_label],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=9,
        )
        for bucket_label, _ in BUCKET_ANALYSIS_SPECS
    ]
    ax.legend(handles=bucket_handles, title="Bucket", loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Circle = FULL, square = POS8",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    paths = ProjectPaths.default()
    paths.ensure_output_dirs()
    label = "full"
    rng = np.random.default_rng(NULL_MODEL_SEED)

    output_data = paths.output_data_dir
    output_figures = paths.output_figures_dir

    profile_metrics_full = (
        pd.read_csv(output_data / f"profile_metrics_{label}.csv")
        .add_prefix("full_")
        .rename(columns={"full_profile_id": "profile_id"})
    )
    profile_metrics_pos8 = (
        pd.read_csv(output_data / f"profile_metrics_positive_8_{label}.csv")
        .add_prefix("pos8_")
        .rename(columns={"pos8_profile_id": "profile_id"})
    )
    profile_metrics_pos3 = (
        build_pos3_profile_metrics(paths)
        .add_prefix("pos3_")
        .rename(columns={"pos3_profile_id": "profile_id"})
    )
    profile_extremes = pd.read_csv(output_data / f"profile_rating_extremes_profiles_{label}.csv")

    interaction_buckets = assign_percentile_buckets(profile_extremes, "total_received").rename(
        columns={"bucket": "interaction_bucket"}
    )
    high_buckets = assign_percentile_buckets(profile_extremes, "high_received").rename(columns={"bucket": "high_bucket"})

    merged = (
        profile_extremes.merge(profile_metrics_full, on="profile_id", how="left")
        .merge(profile_metrics_pos8, on="profile_id", how="left")
        .merge(profile_metrics_pos3, on="profile_id", how="left")
        .merge(interaction_buckets, on="profile_id", how="left")
        .merge(high_buckets, on="profile_id", how="left")
    )

    correlation_rows: list[dict[str, float | int | str | bool]] = []
    for feature_col, feature_label, family in FEATURE_SPECS:
        values = merged[feature_col]
        zscores = zscore_series(values)
        for group_key, group_label, bucket_col, bucket_label in GROUP_SPECS:
            membership = merged[bucket_col].eq(bucket_label)
            membership_corr, p_value, n_valid = point_biserial_stats(values, membership)
            correlation_rows.append(
                {
                    "feature": feature_col,
                    "feature_label": feature_label,
                    "family": family,
                    "group": group_key,
                    "group_label": group_label,
                    "group_size": int(membership.sum()),
                    "group_share": float(membership.mean()),
                    "n_valid": n_valid,
                    "membership_corr": membership_corr,
                    "p_value": p_value,
                    "mean_zscore_in_group": float(zscores[membership].mean()) if membership.any() else 0.0,
                    "mean_feature_in_group": float(values[membership].mean()) if membership.any() else 0.0,
                }
            )

    correlation_frame = pd.DataFrame(correlation_rows)

    summary_rows: list[dict[str, float | str | bool]] = []
    for feature_col, feature_label, family in FEATURE_SPECS:
        subset = correlation_frame[correlation_frame["feature"] == feature_col]
        interaction_subset = subset[subset["group"].str.startswith("interaction_")]
        high_subset = subset[subset["group"].str.startswith("high_")]

        interaction_mean_corr = float(interaction_subset["membership_corr"].mean())
        high_mean_corr = float(high_subset["membership_corr"].mean())
        interaction_mean_z = float(interaction_subset["mean_zscore_in_group"].mean())
        high_mean_z = float(high_subset["mean_zscore_in_group"].mean())
        same_sign_corr = np.sign(interaction_mean_corr) == np.sign(high_mean_corr)
        same_sign_z = np.sign(interaction_mean_z) == np.sign(high_mean_z)
        interaction_bonferroni_alpha = 0.05 / max(len(interaction_subset), 1)
        high_bonferroni_alpha = 0.05 / max(len(high_subset), 1)

        summary_rows.append(
            {
                "feature": feature_col,
                "feature_label": feature_label,
                "family": family,
                "interaction_mean_corr": interaction_mean_corr,
                "high_mean_corr": high_mean_corr,
                "interaction_mean_z": interaction_mean_z,
                "high_mean_z": high_mean_z,
                "same_sign_corr": bool(same_sign_corr),
                "same_sign_z": bool(same_sign_z),
                "shared_signal_strength": float(min(abs(interaction_mean_corr), abs(high_mean_corr))),
                "interaction_sig_0_05_count": int((interaction_subset["p_value"] < 0.05).sum()),
                "high_sig_0_05_count": int((high_subset["p_value"] < 0.05).sum()),
                "interaction_sig_bonferroni_count": int((interaction_subset["p_value"] < interaction_bonferroni_alpha).sum()),
                "high_sig_bonferroni_count": int((high_subset["p_value"] < high_bonferroni_alpha).sum()),
                "interaction_max_p_value": float(interaction_subset["p_value"].max()),
                "high_max_p_value": float(high_subset["p_value"].max()),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["same_sign_corr", "shared_signal_strength"],
        ascending=[False, False],
    )

    overlap_rows: list[dict[str, float | int | str]] = []
    population_size = int(len(merged))
    for bucket_label, _ in BUCKET_ANALYSIS_SPECS:
        interaction_membership = merged["interaction_bucket"].eq(bucket_label)
        high_membership = merged["high_bucket"].eq(bucket_label)
        interaction_size = int(interaction_membership.sum())
        high_size = int(high_membership.sum())
        intersection = int((interaction_membership & high_membership).sum())
        union = int((interaction_membership | high_membership).sum())
        overlap_rows.append(
            {
                "bucket": bucket_label,
                "interaction_bucket_size": interaction_size,
                "high_bucket_size": high_size,
                "intersection_size": intersection,
                "union_size": union,
                "jaccard_overlap": float(intersection / union) if union else 0.0,
                "share_of_interaction_bucket_overlapping": float(intersection / interaction_size)
                if interaction_size
                else 0.0,
                "share_of_high_bucket_overlapping": float(intersection / high_size)
                if high_size
                else 0.0,
                **overlap_null_stats(
                    population_size=population_size,
                    size_a=interaction_size,
                    size_b=high_size,
                    observed_intersection=intersection,
                    rng=rng,
                ),
            }
        )

    overlap_frame = pd.DataFrame(overlap_rows)

    profile_output = output_data / f"profile_feature_alignment_profiles_{label}.csv"
    corr_output = output_data / f"profile_feature_alignment_correlations_{label}.csv"
    feature_class_output = output_data / f"profile_feature_alignment_feature_classes_{label}.csv"
    summary_output = output_data / f"profile_feature_alignment_summary_{label}.csv"
    overlap_output = output_data / f"profile_feature_alignment_overlap_{label}.csv"

    merged.to_csv(profile_output, index=False)
    correlation_frame.to_csv(corr_output, index=False)
    bucket_pair_frame = build_bucket_pair_frame(correlation_frame)
    selected_points = select_label_points(bucket_pair_frame)
    feature_class_frame = build_feature_class_frame(summary_frame)
    feature_class_frame.to_csv(feature_class_output, index=False)
    summary_frame.to_csv(summary_output, index=False)
    overlap_frame.to_csv(overlap_output, index=False)

    corr_pivot = (
        correlation_frame.pivot(index="feature_label", columns="group_label", values="membership_corr")
        .reindex([label for _, label, _ in FEATURE_SPECS])
        .reindex(columns=[label for _, label, _, _ in GROUP_SPECS])
    )
    zscore_pivot = (
        correlation_frame.pivot(index="feature_label", columns="group_label", values="mean_zscore_in_group")
        .reindex([label for _, label, _ in FEATURE_SPECS])
        .reindex(columns=[label for _, label, _, _ in GROUP_SPECS])
    )

    corr_heatmap_path = output_figures / f"profile_feature_alignment_heatmap_{label}.png"
    zscore_heatmap_path = output_figures / f"profile_feature_alignment_zscore_heatmap_{label}.png"
    raw_consistency_path = output_figures / f"profile_feature_alignment_consistency_raw_{label}.png"
    consistency_path = output_figures / f"profile_feature_alignment_consistency_{label}.png"

    plot_heatmap(
        corr_pivot,
        corr_heatmap_path,
        title="Feature Correlation with Elite Bucket Membership",
        cmap="RdBu_r",
        vmin=-0.65,
        vmax=0.65,
        fmt=".2f",
    )
    plot_heatmap(
        zscore_pivot,
        zscore_heatmap_path,
        title="Feature Z-Score Lift Inside Elite Buckets",
        cmap="RdBu_r",
        vmin=-6.5,
        vmax=6.5,
        fmt=".1f",
    )
    plot_consistency_scatter(bucket_pair_frame, selected_points, raw_consistency_path)
    plot_feature_class_scatter(feature_class_frame, consistency_path)

    print(f"Saved profile feature alignment outputs to {paths.outputs_dir}")
    print(summary_frame.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
