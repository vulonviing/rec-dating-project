from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def clean(text: str) -> str:
    return dedent(text).strip() + "\n"


def md(text: str):
    return nbf.v4.new_markdown_cell(clean(text))


def code(text: str):
    return nbf.v4.new_code_cell(clean(text))


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    path.write_text(nbf.writes(nb), encoding="utf-8")


def notebook_01() -> list:
    return [
        md(
            """
            # 01 | Data Preparation

            This notebook is the starting point of the project.

            Its role is simple:

            1. locate the project and the raw dataset
            2. build or refresh the cached dataset summary
            3. inspect the raw fields and the basic scale of the data
            4. explain the modeling choices that the later notebooks rely on

            If someone wants to reproduce the project step by step, this is the first notebook to run.
            """
        ),
        md(
            """
            ## Workflow Map

            The notebook sequence is now organized as a clean four-step pipeline:

            1. **Data preparation**: understand the raw file and cache the dataset summary
            2. **Exploration**: inspect the first descriptive patterns with tables and plots
            3. **Applications**: answer the substantive network questions with focused analyses
            4. **Final plots for the paper**: collect the paper-facing visuals and reference values

            Throughout the notebooks, we use two interpretation styles:

            - a **technical interpretation** for methodological clarity
            - a **plain-language interpretation** for readers who are less technical
            """
        ),
        code(
            '''
            from __future__ import annotations

            import json
            import subprocess
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Markdown, display

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", 40)
            pd.set_option("display.width", 160)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


            def locate_project_root() -> Path:
                cwd = Path.cwd().resolve()
                for candidate in [cwd, *cwd.parents]:
                    if (candidate / "src" / "rec_dating_project").exists():
                        return candidate
                raise RuntimeError("Could not locate the project root from the current working directory.")


            project_root = locate_project_root()
            sys.path.insert(0, str(project_root / "src"))

            from rec_dating_project import ProjectPaths, RecDatingDataset


            paths = ProjectPaths.default()
            paths.ensure_output_dirs()
            OUTPUT_DATA = paths.output_data_dir
            FORCE_REBUILD = False


            def run_script(script_name: str, *args: str) -> None:
                cmd = [sys.executable, str(paths.scripts_dir / script_name), *args]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=paths.project_root)


            def ensure_dataset_summary(force: bool = False) -> Path:
                summary_path = OUTPUT_DATA / "dataset_summary.json"
                if force or not summary_path.exists():
                    run_script("01_dataset_overview.py")
                return summary_path


            summary_path = ensure_dataset_summary(FORCE_REBUILD)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset = RecDatingDataset(paths.raw_edges_path)

            print(f"Project root: {paths.project_root}")
            print(f"Raw edge file: {paths.raw_edges_path}")
            print(f"Summary cache: {summary_path}")
            '''
        ),
        md(
            """
            ## Raw File Preview

            The raw edge file has exactly three columns and no header row:

            - `rater_id`
            - `profile_id`
            - `rating`

            Each row is one directed rating event from a rater to a profile.
            """
        ),
        code(
            '''
            raw_preview = dataset.read_edges(nrows=8)
            raw_preview
            '''
        ),
        code(
            '''
            field_dictionary = pd.DataFrame(
                [
                    {
                        "field": "rater_id",
                        "technical meaning": "The user who sends the rating.",
                        "plain-language meaning": "Who gave the score.",
                    },
                    {
                        "field": "profile_id",
                        "technical meaning": "The user who receives the rating.",
                        "plain-language meaning": "Who was scored.",
                    },
                    {
                        "field": "rating",
                        "technical meaning": "An integer edge weight on the 1-10 scale.",
                        "plain-language meaning": "The score itself.",
                    },
                ]
            )

            display(field_dictionary)
            '''
        ),
        md(
            """
            ## Basic Dataset Scale

            Before we do any network analysis, we want a compact picture of the dataset size and the score distribution.

            This summary is cached in `outputs/data/dataset_summary.json`, so later notebooks can reuse it without re-reading the full file every time.
            """
        ),
        code(
            '''
            summary_table = pd.DataFrame(
                [
                    {"metric": "edge_count", "value": summary["edge_count"]},
                    {"metric": "unique_raters", "value": summary["unique_raters"]},
                    {"metric": "unique_profiles", "value": summary["unique_profiles"]},
                    {"metric": "unique_users_union", "value": summary["unique_users_union"]},
                    {"metric": "overlapping_user_ids", "value": summary["overlapping_user_ids"]},
                    {"metric": "exclusive_profile_ids", "value": summary["exclusive_profile_ids"]},
                    {"metric": "mean_rating", "value": summary["mean_rating"]},
                ]
            )

            rating_distribution = pd.DataFrame(
                [
                    {"rating": int(rating), "count": int(count)}
                    for rating, count in summary["rating_histogram"].items()
                ]
            ).sort_values("rating")
            rating_distribution["share"] = rating_distribution["count"] / rating_distribution["count"].sum()

            display(summary_table)
            display(rating_distribution)
            '''
        ),
        code(
            '''
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(rating_distribution["rating"], rating_distribution["share"], color="#386641", width=0.8)
            ax.set_title("Rating Distribution in the Raw Dataset")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Share of all edges")
            ax.set_xticks(rating_distribution["rating"])
            ax.grid(alpha=0.25, axis="y")
            plt.tight_layout()
            plt.show()
            '''
        ),
        md(
            """
            ## Why a Role-Based Bipartite Network?

            One modeling choice matters for every later result:

            we do **not** treat the same numeric identifier as a single universal node.

            Instead, we separate the two structural roles:

            - the **rater role**: sending scores
            - the **profile role**: receiving scores

            That choice keeps the directional meaning of the data intact and avoids unsupported identity assumptions across roles.
            """
        ),
        code(
            '''
            role_table = pd.DataFrame(
                [
                    {
                        "quantity": "unique_raters",
                        "value": summary["unique_raters"],
                        "why it matters": "This is the sending side of the network.",
                    },
                    {
                        "quantity": "unique_profiles",
                        "value": summary["unique_profiles"],
                        "why it matters": "This is the receiving side of the network.",
                    },
                    {
                        "quantity": "overlapping_user_ids",
                        "value": summary["overlapping_user_ids"],
                        "why it matters": "Many numeric IDs appear in both roles, so the roles must be separated analytically.",
                    },
                ]
            )

            top_raters = pd.DataFrame(summary["top_raters"]).head(10)
            top_profiles = pd.DataFrame(summary["top_profiles"]).head(10)

            display(role_table)
            display(Markdown("### Top raters by outgoing activity"))
            display(top_raters)
            display(Markdown("### Top profiles by incoming attention"))
            display(top_profiles)
            '''
        ),
        code(
            '''
            display(
                Markdown(
                    f"""
                    ## Interpretation

                    ### Technical interpretation

                    - The dataset is large enough to support full-network analysis: **{summary['edge_count']:,}** edges, **{summary['unique_raters']:,}** raters, and **{summary['unique_profiles']:,}** profiles.
                    - The overlap between `rater_id` and `profile_id` values is substantial (**{summary['overlapping_user_ids']:,}** IDs), which justifies the role-based bipartite representation.
                    - The score distribution is not flat; the upper end of the scale is used heavily, which matters for the later positive-layer analysis.

                    ### Plain-language interpretation

                    - This is a very large rating network, not a toy example.
                    - Many users appear in both roles, so it would be misleading to mix "giving ratings" and "receiving ratings" into a single role.
                    - People use high scores quite often, so it makes sense later to ask what happens when we focus on strong positive ratings.
                    """
                )
            )
            '''
        ),
        md(
            """
            ## What This Notebook Produced

            After running this notebook, the key cached artifact is:

            - `outputs/data/dataset_summary.json`

            The next notebook uses that summary together with the first study outputs to explore popularity, prestige, and inequality with tables and figures.
            """
        ),
    ]


def notebook_02() -> list:
    return [
        md(
            """
            # 02 | Exploration

            This notebook is the exploratory stage of the project.

            It answers the first descriptive questions:

            - How do popularity and prestige relate?
            - How unequal is attention on the profile side?
            - Which profiles and raters stand out most strongly?

            The purpose here is not yet to make the most specific substantive claim.
            Instead, we build intuition with tables, rankings, and plots.
            """
        ),
        md(
            """
            ## Notebook Logic

            This notebook uses the first-study pipeline.

            If the required artifacts are missing, it can rebuild them by running the project analysis script.
            By default, it reuses existing outputs to keep the notebook fast.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import json
            import subprocess
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Image, Markdown, display

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", 40)
            pd.set_option("display.width", 160)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


            def locate_project_root() -> Path:
                cwd = Path.cwd().resolve()
                for candidate in [cwd, *cwd.parents]:
                    if (candidate / "src" / "rec_dating_project").exists():
                        return candidate
                raise RuntimeError("Could not locate the project root from the current working directory.")


            project_root = locate_project_root()
            sys.path.insert(0, str(project_root / "src"))

            from rec_dating_project import PopularityPrestigeAnalyzer, ProjectPaths


            paths = ProjectPaths.default()
            paths.ensure_output_dirs()
            OUTPUT_DATA = paths.output_data_dir
            OUTPUT_FIGURES = paths.output_figures_dir
            FORCE_REBUILD = False


            def run_script(script_name: str, *args: str) -> None:
                cmd = [sys.executable, str(paths.scripts_dir / script_name), *args]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=paths.project_root)


            def ensure_dataset_summary(force: bool = False) -> None:
                summary_path = OUTPUT_DATA / "dataset_summary.json"
                if force or not summary_path.exists():
                    run_script("01_dataset_overview.py")


            def ensure_first_study_outputs(force: bool = False) -> None:
                ensure_dataset_summary(force=force)
                required_data = [
                    OUTPUT_DATA / "dataset_summary.json",
                    OUTPUT_DATA / "rating_distribution_full.csv",
                    OUTPUT_DATA / "profile_metrics_full.csv",
                    OUTPUT_DATA / "profile_metrics_positive_8_full.csv",
                    OUTPUT_DATA / "rater_metrics_full.csv",
                    OUTPUT_DATA / "top_profiles_popularity_full.csv",
                    OUTPUT_DATA / "top_profiles_prestige_full.csv",
                    OUTPUT_DATA / "top_profiles_prestige_gap_positive_full.csv",
                    OUTPUT_DATA / "top_profiles_prestige_gap_negative_full.csv",
                    OUTPUT_DATA / "top_raters_activity_full.csv",
                    OUTPUT_DATA / "top_raters_hub_full.csv",
                ]
                required_figures = [
                    OUTPUT_FIGURES / "rating_distribution_full.png",
                    OUTPUT_FIGURES / "strength_ccdf_full.png",
                    OUTPUT_FIGURES / "popularity_vs_prestige_full.png",
                    OUTPUT_FIGURES / "popularity_vs_prestige_2_full.png",
                    OUTPUT_FIGURES / "profile_lorenz_full.png",
                ]
                missing = [path for path in required_data + required_figures if not path.exists()]
                if force or missing:
                    run_script("04_full_project_analysis.py")


            ensure_first_study_outputs(FORCE_REBUILD)

            summary = json.loads((OUTPUT_DATA / "dataset_summary.json").read_text(encoding="utf-8"))
            rating_distribution = pd.read_csv(OUTPUT_DATA / "rating_distribution_full.csv")
            profile_metrics = pd.read_csv(OUTPUT_DATA / "profile_metrics_full.csv")
            positive_profile_metrics = pd.read_csv(OUTPUT_DATA / "profile_metrics_positive_8_full.csv")
            rater_metrics = pd.read_csv(OUTPUT_DATA / "rater_metrics_full.csv")
            top_popularity = pd.read_csv(OUTPUT_DATA / "top_profiles_popularity_full.csv")
            top_prestige = pd.read_csv(OUTPUT_DATA / "top_profiles_prestige_full.csv")
            top_gap_positive = pd.read_csv(OUTPUT_DATA / "top_profiles_prestige_gap_positive_full.csv")
            top_gap_negative = pd.read_csv(OUTPUT_DATA / "top_profiles_prestige_gap_negative_full.csv")
            top_raters_activity = pd.read_csv(OUTPUT_DATA / "top_raters_activity_full.csv")
            top_raters_hub = pd.read_csv(OUTPUT_DATA / "top_raters_hub_full.csv")


            def top_overlap(frame: pd.DataFrame, col_a: str, col_b: str, k: int = 100) -> dict[str, float]:
                a = set(frame.sort_values(col_a, ascending=False).head(k)["profile_id"].tolist())
                b = set(frame.sort_values(col_b, ascending=False).head(k)["profile_id"].tolist())
                inter = len(a & b)
                union = len(a | b)
                return {"k": k, "intersection": inter, "jaccard": (inter / union) if union else 0.0}


            full_corr = {
                "pearson": float(profile_metrics["in_strength"].corr(profile_metrics["authority_score"], method="pearson")),
                "spearman": float(profile_metrics["in_strength"].corr(profile_metrics["authority_score"], method="spearman")),
            }
            pos_corr = {
                "pearson": float(positive_profile_metrics["in_strength"].corr(positive_profile_metrics["authority_score"], method="pearson")),
                "spearman": float(positive_profile_metrics["in_strength"].corr(positive_profile_metrics["authority_score"], method="spearman")),
            }
            overlap = top_overlap(profile_metrics, "in_strength", "authority_score", k=100)
            profile_ineq = {
                "gini_in_strength": PopularityPrestigeAnalyzer.gini(profile_metrics["in_strength"]),
                "gini_authority": PopularityPrestigeAnalyzer.gini(profile_metrics["authority_score"]),
                "top_1pct_in_strength_share": PopularityPrestigeAnalyzer.top_share(profile_metrics["in_strength"], fraction=0.01),
                "top_1pct_authority_share": PopularityPrestigeAnalyzer.top_share(profile_metrics["authority_score"], fraction=0.01),
            }
            '''
        ),
        md(
            """
            ## Core Exploration Summary

            The table below collects the main descriptive numbers from the first study.
            """
        ),
        code(
            '''
            exploration_summary = pd.DataFrame(
                [
                    {"metric": "Full-layer Pearson correlation", "value": full_corr["pearson"]},
                    {"metric": "Full-layer Spearman correlation", "value": full_corr["spearman"]},
                    {"metric": "Positive-layer Pearson correlation (rating >= 8)", "value": pos_corr["pearson"]},
                    {"metric": "Positive-layer Spearman correlation (rating >= 8)", "value": pos_corr["spearman"]},
                    {"metric": "Top-100 popularity/prestige overlap", "value": overlap["intersection"]},
                    {"metric": "Top-100 popularity/prestige Jaccard", "value": overlap["jaccard"]},
                    {"metric": "Gini of profile in-strength", "value": profile_ineq["gini_in_strength"]},
                    {"metric": "Gini of profile authority", "value": profile_ineq["gini_authority"]},
                    {"metric": "Top 1% share of in-strength", "value": profile_ineq["top_1pct_in_strength_share"]},
                    {"metric": "Top 1% share of authority", "value": profile_ineq["top_1pct_authority_share"]},
                ]
            )

            display(exploration_summary)
            '''
        ),
        md(
            """
            ## Visual Exploration

            We now inspect the key figures one by one.

            The first two plots show how the rating scale is used and how heavy-tailed the interaction distribution is.
            The next three focus on popularity, prestige, and inequality on the profile side.
            """
        ),
        code(
            '''
            figure_notes = [
                (
                    "rating_distribution_full.png",
                    "The rating scale is used unevenly, with substantial mass near the upper end.",
                ),
                (
                    "strength_ccdf_full.png",
                    "Interaction is heavy-tailed on both sides of the bipartite graph, especially on the profile side.",
                ),
                (
                    "popularity_vs_prestige_full.png",
                    "The raw scatter shows that more popular profiles are usually also more prestigious.",
                ),
                (
                    "popularity_vs_prestige_2_full.png",
                    "The percentile-binned version gives a cleaner summary of the same relationship.",
                ),
                (
                    "profile_lorenz_full.png",
                    "Both attention and prestige are highly unequal, and raw attention is slightly more concentrated than authority in the summary statistics.",
                ),
            ]

            for filename, note in figure_notes:
                display(Markdown(f"### {filename}\\n\\n{note}"))
                display(Image(filename=str(OUTPUT_FIGURES / filename)))
            '''
        ),
        md(
            """
            ## Concrete Rankings

            Exploratory analysis should not stop at correlations.
            The tables below show which concrete profiles and raters occupy the most visible positions in the network.
            """
        ),
        code(
            '''
            display(Markdown("### Most popular profiles"))
            display(top_popularity.head(10))

            display(Markdown("### Most prestigious profiles"))
            display(top_prestige.head(10))

            display(Markdown("### Profiles whose prestige is higher than their popularity rank would suggest"))
            display(top_gap_positive.head(10))

            display(Markdown("### Profiles whose popularity is higher than their prestige rank would suggest"))
            display(top_gap_negative.head(10))

            display(Markdown("### Most active raters"))
            display(top_raters_activity.head(10))

            display(Markdown("### Strongest hub raters"))
            display(top_raters_hub.head(10))
            '''
        ),
        code(
            '''
            display(
                Markdown(
                    f"""
                    ## Interpretation

                    ### Technical interpretation

                    - Popularity and prestige are strongly aligned in the full network (**Pearson = {full_corr['pearson']:.3f}**, **Spearman = {full_corr['spearman']:.3f}**).
                    - The alignment remains strong when we restrict the graph to positive ratings of at least 8.
                    - The rankings are not identical: the top-100 sets overlap on **{overlap['intersection']}** profiles, not all 100.
                    - Both attention and prestige are highly unequal; the inequality summary suggests that **raw attention is slightly more concentrated than authority** in the full-layer outputs.

                    ### Plain-language interpretation

                    - Profiles that attract a lot of total attention are usually also the profiles that look important in the network structure.
                    - Still, the two ideas are not the same thing. Some profiles are helped or hurt by *who* rates them, not just by *how many* ratings they get.
                    - A relatively small part of the network captures a very large share of attention.
                    """
                )
            )
            '''
        ),
        md(
            """
            ## What Comes Next

            Exploration gives us the big picture.

            The next notebook moves from broad description to **applications**:

            - concentration of high and low ratings
            - concentration of overall interaction
            - feature alignment between the popularity/prestige study and the concentration study
            """
        ),
    ]


def notebook_03() -> list:
    return [
        md(
            """
            # 03 | Applications

            This notebook applies the project framework to the main substantive questions.

            The focus is no longer just "what does the network look like?"
            Instead, we ask:

            - Which profiles concentrate high ratings?
            - Which profiles concentrate low ratings?
            - How does the concentration story connect back to the popularity/prestige features from the first study?
            """
        ),
        md(
            """
            ## Notebook Logic

            This notebook depends on two analysis stages:

            - the **rating-extremes stage**
            - the **feature-alignment stage**

            If the required artifacts are missing, the notebook can rebuild them in sequence.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import subprocess
            import sys
            from pathlib import Path

            import pandas as pd
            from IPython.display import Image, Markdown, display

            pd.set_option("display.max_columns", 60)
            pd.set_option("display.width", 180)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


            def locate_project_root() -> Path:
                cwd = Path.cwd().resolve()
                for candidate in [cwd, *cwd.parents]:
                    if (candidate / "src" / "rec_dating_project").exists():
                        return candidate
                raise RuntimeError("Could not locate the project root from the current working directory.")


            project_root = locate_project_root()
            sys.path.insert(0, str(project_root / "src"))

            from rec_dating_project import ProjectPaths


            paths = ProjectPaths.default()
            paths.ensure_output_dirs()
            OUTPUT_DATA = paths.output_data_dir
            OUTPUT_FIGURES = paths.output_figures_dir
            FORCE_REBUILD = False


            def run_script(script_name: str, *args: str) -> None:
                cmd = [sys.executable, str(paths.scripts_dir / script_name), *args]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=paths.project_root)


            def ensure_stage_two_outputs(force: bool = False) -> None:
                required = [
                    OUTPUT_DATA / "profile_metrics_full.csv",
                    OUTPUT_DATA / "profile_metrics_positive_8_full.csv",
                    OUTPUT_DATA / "rater_metrics_full.csv",
                ]
                if force or any(not path.exists() for path in required):
                    run_script("04_full_project_analysis.py")


            def ensure_stage_three_outputs(force: bool = False) -> None:
                ensure_stage_two_outputs(force=force)

                extremes_required = [
                    OUTPUT_DATA / "profile_rating_extremes_summary_full.csv",
                    OUTPUT_DATA / "profile_rating_extremes_buckets_full.csv",
                    OUTPUT_DATA / "profile_rating_extremes_profiles_full.csv",
                    OUTPUT_DATA / "top_profiles_high_ratings_full.csv",
                    OUTPUT_DATA / "top_profiles_low_ratings_full.csv",
                    OUTPUT_FIGURES / "profile_high_low_rating_concentration_full.png",
                    OUTPUT_FIGURES / "profile_rating_concentration_curves_full.png",
                    OUTPUT_FIGURES / "profile_bucket_shares_full.png",
                    OUTPUT_FIGURES / "profile_interaction_concentration_curve_full.png",
                    OUTPUT_FIGURES / "profile_interaction_bucket_shares_full.png",
                ]
                if force or any(not path.exists() for path in extremes_required):
                    run_script("07_profile_rating_extremes.py")

                alignment_required = [
                    OUTPUT_DATA / "profile_feature_alignment_summary_full.csv",
                    OUTPUT_DATA / "profile_feature_alignment_overlap_full.csv",
                    OUTPUT_DATA / "profile_feature_alignment_correlations_full.csv",
                    OUTPUT_DATA / "profile_feature_alignment_profiles_full.csv",
                    OUTPUT_DATA / "profile_feature_alignment_feature_classes_full.csv",
                    OUTPUT_FIGURES / "profile_feature_alignment_heatmap_full.png",
                    OUTPUT_FIGURES / "profile_feature_alignment_zscore_heatmap_full.png",
                    OUTPUT_FIGURES / "profile_feature_alignment_consistency_full.png",
                ]
                if force or any(not path.exists() for path in alignment_required):
                    run_script("08_profile_feature_alignment.py")


            ensure_stage_three_outputs(FORCE_REBUILD)

            extreme_summary = pd.read_csv(OUTPUT_DATA / "profile_rating_extremes_summary_full.csv")
            extreme_buckets = pd.read_csv(OUTPUT_DATA / "profile_rating_extremes_buckets_full.csv")
            top_high = pd.read_csv(OUTPUT_DATA / "top_profiles_high_ratings_full.csv")
            top_low = pd.read_csv(OUTPUT_DATA / "top_profiles_low_ratings_full.csv")

            alignment_summary = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_summary_full.csv")
            alignment_overlap = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_overlap_full.csv")
            alignment_correlations = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_correlations_full.csv")
            alignment_profiles = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_profiles_full.csv")
            alignment_feature_classes = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_feature_classes_full.csv")

            all_summary = extreme_summary.loc[extreme_summary["series"] == "all_interactions"].iloc[0]
            high_summary = extreme_summary.loc[extreme_summary["series"] == "high_ratings"].iloc[0]
            low_summary = extreme_summary.loc[extreme_summary["series"] == "low_ratings"].iloc[0]

            bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
            rating_bucket_view = (
                extreme_buckets.pivot(index="bucket", columns="series", values="value_share")
                .reindex(bucket_order)
                .rename(columns={"high_ratings": "high_ratings_share", "low_ratings": "low_ratings_share"})
                [["high_ratings_share", "low_ratings_share"]]
                .reset_index()
            )
            interaction_bucket_view = (
                extreme_buckets.loc[extreme_buckets["series"] == "all_interactions", ["bucket", "value_share"]]
                .rename(columns={"value_share": "all_interactions_share"})
                .set_index("bucket")
                .reindex(bucket_order)
                .reset_index()
            )
            '''
        ),
        md(
            """
            ## Application 1 | Rating Extremes and Interaction Concentration

            We start with the profile side:

            - overall interaction concentration
            - high-rating concentration
            - low-rating concentration

            This tells us whether the same small subset of profiles dominates multiple kinds of received attention.
            """
        ),
        code(
            '''
            concentration_summary = pd.DataFrame(
                [
                    {"series": "all_interactions", "top_1pct_share": all_summary["top_1pct_share"], "top_10pct_share": all_summary["top_10pct_share"], "top_20pct_share": all_summary["top_20pct_share"], "user_share_for_80pct": all_summary["user_share_for_80pct"]},
                    {"series": "high_ratings", "top_1pct_share": high_summary["top_1pct_share"], "top_10pct_share": high_summary["top_10pct_share"], "top_20pct_share": high_summary["top_20pct_share"], "user_share_for_80pct": high_summary["user_share_for_80pct"]},
                    {"series": "low_ratings", "top_1pct_share": low_summary["top_1pct_share"], "top_10pct_share": low_summary["top_10pct_share"], "top_20pct_share": low_summary["top_20pct_share"], "user_share_for_80pct": low_summary["user_share_for_80pct"]},
                ]
            )

            display(concentration_summary)
            display(rating_bucket_view)
            display(interaction_bucket_view)
            '''
        ),
        code(
            '''
            concentration_figures = [
                (
                    "profile_high_low_rating_concentration_full.png",
                    "Cumulative concentration of high and low ratings.",
                ),
                (
                    "profile_rating_concentration_curves_full.png",
                    "Separate concentration curves for high and low ratings.",
                ),
                (
                    "profile_bucket_shares_full.png",
                    "Disjoint bucket shares for high and low ratings.",
                ),
                (
                    "profile_interaction_concentration_curve_full.png",
                    "Overall interaction concentration on the profile side.",
                ),
                (
                    "profile_interaction_bucket_shares_full.png",
                    "Disjoint bucket shares for overall interaction.",
                ),
            ]

            for filename, note in concentration_figures:
                display(Markdown(f"### {filename}\\n\\n{note}"))
                display(Image(filename=str(OUTPUT_FIGURES / filename)))
            '''
        ),
        code(
            '''
            display(Markdown("### Profiles receiving the most high ratings"))
            display(top_high.head(15))

            display(Markdown("### Profiles receiving the most low ratings"))
            display(top_low.head(15))

            display(
                Markdown(
                    f"""
                    ## Interpretation of Application 1

                    ### Technical interpretation

                    - The top **1%** of profiles receives **{all_summary['top_1pct_share']:.2%}** of all interactions.
                    - The same top **1%** receives **{high_summary['top_1pct_share']:.2%}** of all high ratings.
                    - Low ratings are concentrated too, but less sharply than high ratings.
                    - This means that concentration is a general property of the profile side, not only a property of positive ratings.

                    ### Plain-language interpretation

                    - A very small set of profiles gets a huge share of attention.
                    - This is even more obvious when we focus only on strong positive ratings.
                    - Negative ratings are also not spread evenly, which suggests that visible profiles attract both approval and criticism.
                    """
                )
            )
            '''
        ),
        md(
            """
            ## Application 2 | Feature Alignment Across the Full Bucket Ladder

            The next question is whether the concentration study and the popularity/prestige study tell a connected story.

            We compare the exact profile buckets from the concentration analysis with the node-level features from the first study.
            """
        ),
        code(
            '''
            overlap_view = alignment_overlap.copy()
            overlap_view["bucket"] = pd.Categorical(overlap_view["bucket"], categories=bucket_order, ordered=True)
            overlap_view = overlap_view.sort_values("bucket")
            display(overlap_view.round(4))

            top_overlap = alignment_overlap.loc[alignment_overlap["bucket"] == "Top 1%"].iloc[0]
            bottom_overlap = alignment_overlap.loc[alignment_overlap["bucket"] == "Bottom 50%"].iloc[0]

            display(
                Markdown(
                    f"""
                    ### Exact-bucket overlap summary

                    - Top 1% overlap: **{top_overlap['intersection_size']:,} shared profiles**, Jaccard **{top_overlap['jaccard_overlap']:.2%}**
                    - Bottom 50% overlap: **{bottom_overlap['intersection_size']:,} shared profiles**, Jaccard **{bottom_overlap['jaccard_overlap']:.2%}**
                    - Random-null expectation for the Top 1% overlap: **{top_overlap['expected_random_intersection']:.1f}** profiles
                    """
                )
            )
            '''
        ),
        code(
            '''
            alignment_figures = [
                (
                    "profile_feature_alignment_heatmap_full.png",
                    "Point-biserial correlations between Study 1 features and exact bucket membership.",
                ),
                (
                    "profile_feature_alignment_zscore_heatmap_full.png",
                    "Standardized feature lift inside each exact bucket.",
                ),
                (
                    "profile_feature_alignment_consistency_full.png",
                    "Class-level alignment across ALL, POS8, and POS3 layers.",
                ),
            ]

            for filename, note in alignment_figures:
                display(Markdown(f"### {filename}\\n\\n{note}"))
                display(Image(filename=str(OUTPUT_FIGURES / filename)))
            '''
        ),
        code(
            '''
            class_view = alignment_feature_classes[
                [
                    "class_label",
                    "layer_label",
                    "feature_count",
                    "interaction_mean_corr",
                    "high_mean_corr",
                    "included_features",
                ]
            ].sort_values(["class_label", "layer_label"])

            positive_signals = alignment_summary[
                (alignment_summary["interaction_mean_corr"] > 0) & (alignment_summary["high_mean_corr"] > 0)
            ].sort_values("shared_signal_strength", ascending=False)

            display(Markdown("### Class summary"))
            display(class_view.round(3))

            display(Markdown("### Strongest shared positive signals"))
            display(
                positive_signals[
                    [
                        "feature_label",
                        "interaction_mean_corr",
                        "high_mean_corr",
                        "shared_signal_strength",
                    ]
                ].head(8).round(3)
            )
            '''
        ),
        code(
            '''
            display(
                Markdown(
                    f"""
                    ## Interpretation of Application 2

                    ### Technical interpretation

                    - The alignment is strongest at the extremes of the ranking, especially in the Top 1% and Bottom 50% buckets.
                    - The observed Top 1% overlap (**{top_overlap['intersection_size']:,}** profiles) is far larger than the fixed-size random expectation (**{top_overlap['expected_random_intersection']:.1f}**).
                    - The most stable shared signals come from degree, strength, authority, and rank-based variables.

                    ### Plain-language interpretation

                    - The profiles that dominate general attention are often the same profiles that dominate strong positive ratings.
                    - This is not a random coincidence.
                    - The first study and the concentration study therefore support each other: they are two views of the same broad hierarchy.
                    """
                )
            )
            '''
        ),
        md(
            """
            ## Hand-off to the Final Notebook

            At this point, the project has already produced the main analytical results.

            The final notebook does one narrower job:

            - collect the figures used in the paper
            - present paper-facing reference values
            - make it easier to compare notebook outputs with the written paper
            """
        ),
    ]


def notebook_04() -> list:
    return [
        md(
            """
            # 04 | Final Plots for the Paper

            This notebook is the paper-facing endpoint of the workflow.

            It does three things:

            1. ensures that the final paper figures exist
            2. displays those figures in the same narrative order used by the paper
            3. collects the reference values that the paper text should agree with
            """
        ),
        md(
            """
            ## Notebook Logic

            The earlier notebooks produced the study outputs.
            Here we add the final degree-distribution fit if needed and then assemble the paper-ready material.
            """
        ),
        code(
            '''
            from __future__ import annotations

            import json
            import subprocess
            import sys
            from pathlib import Path

            import pandas as pd
            from IPython.display import Image, Markdown, display

            pd.set_option("display.max_columns", 60)
            pd.set_option("display.width", 180)
            pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


            def locate_project_root() -> Path:
                cwd = Path.cwd().resolve()
                for candidate in [cwd, *cwd.parents]:
                    if (candidate / "src" / "rec_dating_project").exists():
                        return candidate
                raise RuntimeError("Could not locate the project root from the current working directory.")


            project_root = locate_project_root()
            sys.path.insert(0, str(project_root / "src"))

            from rec_dating_project import PopularityPrestigeAnalyzer, ProjectPaths


            paths = ProjectPaths.default()
            paths.ensure_output_dirs()
            OUTPUT_DATA = paths.output_data_dir
            OUTPUT_FIGURES = paths.output_figures_dir
            FORCE_REBUILD = False


            def run_script(script_name: str, *args: str) -> None:
                cmd = [sys.executable, str(paths.scripts_dir / script_name), *args]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True, cwd=paths.project_root)


            def ensure_all_outputs(force: bool = False) -> None:
                stage_two = [
                    OUTPUT_DATA / "profile_metrics_full.csv",
                    OUTPUT_DATA / "profile_metrics_positive_8_full.csv",
                    OUTPUT_DATA / "rater_metrics_full.csv",
                    OUTPUT_DATA / "rating_distribution_full.csv",
                    OUTPUT_FIGURES / "rating_distribution_full.png",
                    OUTPUT_FIGURES / "strength_ccdf_full.png",
                    OUTPUT_FIGURES / "popularity_vs_prestige_2_full.png",
                    OUTPUT_FIGURES / "profile_lorenz_full.png",
                ]
                if force or any(not path.exists() for path in stage_two):
                    run_script("04_full_project_analysis.py")

                stage_three = [
                    OUTPUT_DATA / "profile_rating_extremes_summary_full.csv",
                    OUTPUT_DATA / "profile_feature_alignment_overlap_full.csv",
                    OUTPUT_DATA / "profile_feature_alignment_feature_classes_full.csv",
                    OUTPUT_FIGURES / "profile_feature_alignment_consistency_full.png",
                ]
                if force or any(not path.exists() for path in stage_three):
                    run_script("07_profile_rating_extremes.py")
                    run_script("08_profile_feature_alignment.py")

                degree_fit = [
                    OUTPUT_DATA / "degree_distribution_fit_full.json",
                    OUTPUT_FIGURES / "degree_distribution_fit_full.png",
                ]
                if force or any(not path.exists() for path in degree_fit):
                    run_script("09_degree_distribution_fit.py")


            ensure_all_outputs(FORCE_REBUILD)

            profile_metrics = pd.read_csv(OUTPUT_DATA / "profile_metrics_full.csv")
            positive_profile_metrics = pd.read_csv(OUTPUT_DATA / "profile_metrics_positive_8_full.csv")
            degree_fit = json.loads((OUTPUT_DATA / "degree_distribution_fit_full.json").read_text(encoding="utf-8"))
            concentration_summary = pd.read_csv(OUTPUT_DATA / "profile_rating_extremes_summary_full.csv")
            concentration_buckets = pd.read_csv(OUTPUT_DATA / "profile_rating_extremes_buckets_full.csv")
            alignment_overlap = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_overlap_full.csv")
            alignment_feature_classes = pd.read_csv(OUTPUT_DATA / "profile_feature_alignment_feature_classes_full.csv")


            def top_overlap(frame: pd.DataFrame, col_a: str, col_b: str, k: int = 100) -> dict[str, float]:
                a = set(frame.sort_values(col_a, ascending=False).head(k)["profile_id"].tolist())
                b = set(frame.sort_values(col_b, ascending=False).head(k)["profile_id"].tolist())
                inter = len(a & b)
                union = len(a | b)
                return {"k": k, "intersection": inter, "jaccard": (inter / union) if union else 0.0}


            full_corr = {
                "pearson": float(profile_metrics["in_strength"].corr(profile_metrics["authority_score"], method="pearson")),
                "spearman": float(profile_metrics["in_strength"].corr(profile_metrics["authority_score"], method="spearman")),
            }
            pos_corr = {
                "pearson": float(positive_profile_metrics["in_strength"].corr(positive_profile_metrics["authority_score"], method="pearson")),
                "spearman": float(positive_profile_metrics["in_strength"].corr(positive_profile_metrics["authority_score"], method="spearman")),
            }
            overlap = top_overlap(profile_metrics, "in_strength", "authority_score", k=100)
            gini_in_strength = PopularityPrestigeAnalyzer.gini(profile_metrics["in_strength"])
            gini_authority = PopularityPrestigeAnalyzer.gini(profile_metrics["authority_score"])

            bucket_order = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-20%", "Top 20-50%", "Bottom 50%"]
            bucket_view = concentration_buckets.pivot(index="bucket", columns="series", values="value_share").reindex(bucket_order)
            top1_overlap = alignment_overlap.loc[alignment_overlap["bucket"] == "Top 1%"].iloc[0]
            '''
        ),
        md(
            """
            ## Paper Figure 1 | Popularity and Prestige Alignment
            """
        ),
        code(
            '''
            display(
                Markdown(
                    """
                    **Why this figure matters**

                    This is the cleanest visual summary of the first study.
                    It shows that popularity and prestige move together strongly even when we avoid the raw log-log scatter.
                    """
                )
            )
            display(Image(filename=str(OUTPUT_FIGURES / "popularity_vs_prestige_2_full.png")))
            '''
        ),
        md(
            """
            ## Paper Figures 2–4 | Concentration and Degree Distribution
            """
        ),
        code(
            '''
            concentration_gallery = [
                ("strength_ccdf_full.png", "Heavy-tailed interaction on the profile and rater sides."),
                (
                    "degree_distribution_fit_full.png",
                    f"Profile in-degree with a fitted power-law tail: alpha = {degree_fit['power_law_alpha']:.2f}, x_min = {int(degree_fit['power_law_xmin'])}.",
                ),
                ("profile_lorenz_full.png", "Lorenz curves for in-strength and authority on the profile side."),
            ]

            for filename, note in concentration_gallery:
                display(Markdown(f"### {filename}\\n\\n{note}"))
                display(Image(filename=str(OUTPUT_FIGURES / filename)))
            '''
        ),
        md(
            """
            ## Paper Figure 5 | Feature Alignment
            """
        ),
        code(
            '''
            display(
                Markdown(
                    """
                    **Why this figure matters**

                    This figure links the first study and the concentration study.
                    It asks whether broader feature families move in compatible directions for both interaction buckets and high-rating buckets.
                    """
                )
            )
            display(Image(filename=str(OUTPUT_FIGURES / "profile_feature_alignment_consistency_full.png")))
            '''
        ),
        md(
            """
            ## Paper-Facing Reference Values

            The table below collects the values that the paper text should agree with.
            """
        ),
        code(
            '''
            reference_values = pd.DataFrame(
                [
                    {"section": "Popularity vs prestige", "metric": "Pearson correlation", "value": full_corr["pearson"]},
                    {"section": "Popularity vs prestige", "metric": "Spearman correlation", "value": full_corr["spearman"]},
                    {"section": "Popularity vs prestige", "metric": "Top-100 overlap", "value": overlap["intersection"]},
                    {"section": "Popularity vs prestige", "metric": "Top-100 Jaccard", "value": overlap["jaccard"]},
                    {"section": "Positive layer", "metric": "Pearson correlation", "value": pos_corr["pearson"]},
                    {"section": "Positive layer", "metric": "Spearman correlation", "value": pos_corr["spearman"]},
                    {"section": "Concentration", "metric": "Gini of in-strength", "value": gini_in_strength},
                    {"section": "Concentration", "metric": "Gini of authority", "value": gini_authority},
                    {"section": "Concentration", "metric": "Top 1% share of all interactions", "value": bucket_view.loc["Top 1%", "all_interactions"]},
                    {"section": "Concentration", "metric": "Top 1% share of high ratings", "value": bucket_view.loc["Top 1%", "high_ratings"]},
                    {"section": "Concentration", "metric": "Top 1% share of low ratings", "value": bucket_view.loc["Top 1%", "low_ratings"]},
                    {"section": "Feature alignment", "metric": "Top 1% overlap count", "value": top1_overlap["intersection_size"]},
                    {"section": "Feature alignment", "metric": "Top 1% overlap Jaccard", "value": top1_overlap["jaccard_overlap"]},
                    {"section": "Feature alignment", "metric": "Top 1% expected random overlap", "value": top1_overlap["expected_random_intersection"]},
                    {"section": "Degree fit", "metric": "Power-law alpha", "value": degree_fit["power_law_alpha"]},
                    {"section": "Degree fit", "metric": "Power-law x_min", "value": degree_fit["power_law_xmin"]},
                    {"section": "Degree fit", "metric": "Nodes in tail", "value": degree_fit["n_in_tail"]},
                    {"section": "Degree fit", "metric": "Power-law vs log-normal R", "value": degree_fit["vs_lognormal_R"]},
                    {"section": "Degree fit", "metric": "Power-law vs exponential R", "value": degree_fit["vs_exponential_R"]},
                ]
            )

            display(reference_values)
            '''
        ),
        code(
            '''
            concentration_direction = "raw attention is slightly more concentrated than authority" if gini_in_strength > gini_authority else "authority is slightly more concentrated than raw attention"

            display(
                Markdown(
                    f"""
                    ## Interpretation

                    ### Technical interpretation

                    - The paper-facing figures support the same broad story as the analysis notebooks.
                    - Popularity and prestige are strongly aligned, but not identical.
                    - The profile side is extremely concentrated.
                    - In the current outputs, **{concentration_direction}**.
                    - The Top 1% overlap between interaction and high-rating buckets is **{top1_overlap['intersection_size']:,}** profiles, far above the random baseline of **{top1_overlap['expected_random_intersection']:.1f}**.

                    ### Plain-language interpretation

                    - A small group of profiles dominates attention.
                    - The same small group also dominates strong positive ratings much more than chance would predict.
                    - These paper figures are therefore not separate stories; together they describe the same hierarchy from different angles.
                    """
                )
            )
            '''
        ),
        md(
            """
            ## Final Note

            This notebook does not edit the paper automatically.
            Its purpose is to make the paper-facing figures and reference values easy to inspect before the paper text is finalized.
            """
        ),
    ]


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    write_notebook(NOTEBOOKS_DIR / "01_data_preparation.ipynb", notebook_01())
    write_notebook(NOTEBOOKS_DIR / "02_rec_dating_exploration.ipynb", notebook_02())
    write_notebook(NOTEBOOKS_DIR / "03_applications.ipynb", notebook_03())
    write_notebook(NOTEBOOKS_DIR / "04_final_plots_for_paper.ipynb", notebook_04())

    obsolete = [
        NOTEBOOKS_DIR / "03_final_project_analysis.ipynb",
        NOTEBOOKS_DIR / "04_role_concentration_analysis.ipynb",
        NOTEBOOKS_DIR / "05_plots_and_integrated_analysis.ipynb",
        NOTEBOOKS_DIR / "06_feature_alignment_analysis.ipynb",
    ]
    for path in obsolete:
        if path.exists():
            path.unlink()

    print("Rebuilt notebook workflow:")
    for name in [
        "01_data_preparation.ipynb",
        "02_rec_dating_exploration.ipynb",
        "03_applications.ipynb",
        "04_final_plots_for_paper.ipynb",
    ]:
        print(f"  - {NOTEBOOKS_DIR / name}")


if __name__ == "__main__":
    main()
