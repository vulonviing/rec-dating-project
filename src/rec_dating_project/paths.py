from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    outputs_dir: Path
    output_data_dir: Path
    output_figures_dir: Path
    output_reports_dir: Path
    notebooks_dir: Path
    scripts_dir: Path
    src_dir: Path

    @classmethod
    def default(cls) -> "ProjectPaths":
        project_root = Path(__file__).resolve().parents[2]
        return cls(
            project_root=project_root,
            data_dir=project_root / "data",
            raw_dir=project_root / "data" / "rec-dating",
            outputs_dir=project_root / "outputs",
            output_data_dir=project_root / "outputs" / "data",
            output_figures_dir=project_root / "outputs" / "figures",
            output_reports_dir=project_root / "outputs" / "reports",
            notebooks_dir=project_root / "notebooks",
            scripts_dir=project_root / "scripts",
            src_dir=project_root / "src",
        )

    @property
    def raw_edges_path(self) -> Path:
        return self.raw_dir / "rec-dating.edges"

    @property
    def readme_path(self) -> Path:
        return self.raw_dir / "readme.html"

    def ensure_output_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_figures_dir.mkdir(parents=True, exist_ok=True)
        self.output_reports_dir.mkdir(parents=True, exist_ok=True)
