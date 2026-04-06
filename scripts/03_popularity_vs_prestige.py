from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_dating_project import (
    PopularityPrestigeAnalyzer,
    ProjectPaths,
    RecDatingDataset,
    RoleBasedBipartiteNetwork,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run popularity vs prestige analysis.")
    parser.add_argument("--nrows", type=int, default=500_000, help="How many edges to use.")
    parser.add_argument("--top-k", type=int, default=200, help="How many rows to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = ProjectPaths.default()
    paths.ensure_output_dirs()

    dataset = RecDatingDataset(paths.raw_edges_path)
    network = RoleBasedBipartiteNetwork(dataset)

    summary = dataset.compute_summary(nrows=args.nrows)
    snapshot = network.build_sparse_rating_matrix(nrows=args.nrows, summary=summary)

    analyzer = PopularityPrestigeAnalyzer(snapshot)
    profile_metrics = analyzer.profile_metrics()
    rater_metrics = analyzer.rater_metrics()
    correlations = analyzer.popularity_vs_prestige_correlation(profile_metrics)

    profile_path = paths.output_data_dir / f"profile_metrics_top_{args.top_k}_{args.nrows}.csv"
    rater_path = paths.output_data_dir / f"rater_metrics_top_{args.top_k}_{args.nrows}.csv"
    report_path = paths.output_reports_dir / f"popularity_prestige_report_{args.nrows}.json"

    profile_metrics.head(args.top_k).to_csv(profile_path, index=False)
    rater_metrics.head(args.top_k).to_csv(rater_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "nrows": args.nrows,
                "top_k": args.top_k,
                "hits_iterations": analyzer.compute_hits().iterations,
                "hits_converged": analyzer.compute_hits().converged,
                "correlations": correlations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Top profiles by authority:")
    print(profile_metrics.head(10).to_string(index=False))
    print("\nCorrelation summary:")
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
