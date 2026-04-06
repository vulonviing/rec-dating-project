from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network import BipartiteSnapshot


@dataclass(frozen=True)
class HitsResult:
    hub_scores: np.ndarray
    authority_scores: np.ndarray
    iterations: int
    converged: bool


class PopularityPrestigeAnalyzer:
    def __init__(self, snapshot: BipartiteSnapshot) -> None:
        self.snapshot = snapshot
        self._hits_cache: HitsResult | None = None

    @staticmethod
    def _safe_l2_norm(vector: np.ndarray) -> float:
        return float(np.sqrt(np.square(vector, dtype=np.float64).sum(dtype=np.float64)))

    @staticmethod
    def gini(values: np.ndarray | pd.Series) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return 0.0
        if np.allclose(array, 0.0):
            return 0.0
        sorted_array = np.sort(array)
        n = sorted_array.size
        cumulative = np.cumsum(sorted_array)
        return float((n + 1 - 2 * cumulative.sum() / cumulative[-1]) / n)

    @staticmethod
    def top_share(values: np.ndarray | pd.Series, fraction: float = 0.01) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return 0.0
        k = max(1, int(np.ceil(array.size * fraction)))
        total = array.sum()
        if total == 0:
            return 0.0
        return float(np.sort(array)[-k:].sum() / total)

    @staticmethod
    def fraction_needed_for_share(
        values: np.ndarray | pd.Series,
        target_share: float = 0.8,
    ) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array) & (array >= 0)]
        if array.size == 0:
            return 0.0
        total = array.sum()
        if total == 0:
            return 0.0
        sorted_desc = np.sort(array)[::-1]
        cumulative = np.cumsum(sorted_desc)
        cutoff = float(np.clip(target_share, 0.0, 1.0)) * total
        needed = int(np.searchsorted(cumulative, cutoff, side="left")) + 1
        return float(needed / array.size)

    def compute_hits(
        self,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> HitsResult:
        if self._hits_cache is not None:
            return self._hits_cache

        matrix = self.snapshot.matrix.astype(np.float64, copy=False)
        hub_scores = np.ones(matrix.shape[0], dtype=np.float64)
        authority_scores = np.ones(matrix.shape[1], dtype=np.float64)

        hub_scores /= self._safe_l2_norm(hub_scores)
        authority_scores /= self._safe_l2_norm(authority_scores)

        converged = False
        iterations = 0

        for iteration in range(1, max_iter + 1):
            new_hub_scores = matrix @ authority_scores
            hub_norm = self._safe_l2_norm(new_hub_scores)
            if hub_norm > 0:
                new_hub_scores /= hub_norm
            else:
                break

            new_authority_scores = matrix.T @ new_hub_scores
            authority_norm = self._safe_l2_norm(new_authority_scores)
            if authority_norm > 0:
                new_authority_scores /= authority_norm
            else:
                break

            delta = max(
                self._safe_l2_norm(new_hub_scores - hub_scores),
                self._safe_l2_norm(new_authority_scores - authority_scores),
            )

            hub_scores = new_hub_scores
            authority_scores = new_authority_scores
            iterations = iteration

            if delta < tol:
                converged = True
                break

        result = HitsResult(
            hub_scores=hub_scores,
            authority_scores=authority_scores,
            iterations=iterations,
            converged=converged,
        )
        self._hits_cache = result
        return result

    def profile_metrics(self, active_only: bool = True) -> pd.DataFrame:
        hits = self.compute_hits()
        matrix = self.snapshot.matrix

        in_degree = matrix.getnnz(axis=0)
        in_strength = np.asarray(matrix.sum(axis=0)).ravel()
        mean_rating = np.divide(
            in_strength,
            in_degree,
            out=np.zeros_like(in_strength, dtype=np.float64),
            where=in_degree > 0,
        )

        frame = pd.DataFrame(
            {
                "profile_id": np.arange(1, self.snapshot.num_profiles + 1, dtype=np.int32),
                "in_degree": in_degree.astype(np.int32),
                "in_strength": in_strength.astype(np.float64),
                "mean_rating": mean_rating.astype(np.float64),
                "authority_score": hits.authority_scores.astype(np.float64),
            }
        )
        frame["popularity_rank_pct"] = frame["in_strength"].rank(pct=True, method="average")
        frame["prestige_rank_pct"] = frame["authority_score"].rank(pct=True, method="average")
        frame["prestige_gap"] = frame["prestige_rank_pct"] - frame["popularity_rank_pct"]

        if active_only:
            frame = frame[frame["in_degree"] > 0].copy()

        return frame.sort_values("authority_score", ascending=False).reset_index(drop=True)

    def rater_metrics(self, active_only: bool = True) -> pd.DataFrame:
        hits = self.compute_hits()
        matrix = self.snapshot.matrix

        out_degree = matrix.getnnz(axis=1)
        out_strength = np.asarray(matrix.sum(axis=1)).ravel()
        mean_rating_given = np.divide(
            out_strength,
            out_degree,
            out=np.zeros_like(out_strength, dtype=np.float64),
            where=out_degree > 0,
        )

        frame = pd.DataFrame(
            {
                "rater_id": np.arange(1, self.snapshot.num_raters + 1, dtype=np.int32),
                "out_degree": out_degree.astype(np.int32),
                "out_strength": out_strength.astype(np.float64),
                "mean_rating_given": mean_rating_given.astype(np.float64),
                "hub_score": hits.hub_scores.astype(np.float64),
            }
        )
        frame["activity_rank_pct"] = frame["out_strength"].rank(pct=True, method="average")
        frame["hub_rank_pct"] = frame["hub_score"].rank(pct=True, method="average")
        frame["hub_gap"] = frame["hub_rank_pct"] - frame["activity_rank_pct"]

        if active_only:
            frame = frame[frame["out_degree"] > 0].copy()

        return frame.sort_values("hub_score", ascending=False).reset_index(drop=True)

    def popularity_vs_prestige_correlation(
        self,
        profile_frame: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        frame = profile_frame if profile_frame is not None else self.profile_metrics()
        return {
            "pearson": float(frame["in_strength"].corr(frame["authority_score"], method="pearson")),
            "spearman": float(frame["in_strength"].corr(frame["authority_score"], method="spearman")),
        }

    def profile_inequality_summary(
        self,
        profile_frame: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        frame = profile_frame if profile_frame is not None else self.profile_metrics()
        return {
            "gini_in_degree": self.gini(frame["in_degree"]),
            "gini_in_strength": self.gini(frame["in_strength"]),
            "gini_authority": self.gini(frame["authority_score"]),
            "top_1pct_in_strength_share": self.top_share(frame["in_strength"], fraction=0.01),
            "top_1pct_authority_share": self.top_share(frame["authority_score"], fraction=0.01),
        }

    def rater_inequality_summary(
        self,
        rater_frame: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        frame = rater_frame if rater_frame is not None else self.rater_metrics()
        return {
            "gini_out_degree": self.gini(frame["out_degree"]),
            "gini_out_strength": self.gini(frame["out_strength"]),
            "gini_hub_score": self.gini(frame["hub_score"]),
            "top_1pct_out_strength_share": self.top_share(frame["out_strength"], fraction=0.01),
            "top_1pct_hub_share": self.top_share(frame["hub_score"], fraction=0.01),
        }
