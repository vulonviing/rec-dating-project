from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_dating_project import ProjectPaths, RecDatingDataset


def main() -> None:
    paths = ProjectPaths.default()
    paths.ensure_output_dirs()

    dataset = RecDatingDataset(paths.raw_edges_path)
    summary = dataset.compute_summary()

    print(summary.to_frame().to_string(index=False))

    summary_path = paths.output_data_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
