from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


EDGE_COLUMNS = ["rater_id", "profile_id", "rating"]
EDGE_DTYPES = {"rater_id": "int32", "profile_id": "int32", "rating": "int16"}


@dataclass(frozen=True)
class DatasetSummary:
    edge_count: int
    unique_raters: int
    unique_profiles: int
    unique_users_union: int
    overlapping_user_ids: int
    exclusive_profile_ids: int
    max_rater_id: int
    max_profile_id: int
    mean_rating: float
    rating_histogram: dict[int, int]
    top_raters: list[tuple[int, int]]
    top_profiles: list[tuple[int, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_count": self.edge_count,
            "unique_raters": self.unique_raters,
            "unique_profiles": self.unique_profiles,
            "unique_users_union": self.unique_users_union,
            "overlapping_user_ids": self.overlapping_user_ids,
            "exclusive_profile_ids": self.exclusive_profile_ids,
            "max_rater_id": self.max_rater_id,
            "max_profile_id": self.max_profile_id,
            "mean_rating": self.mean_rating,
            "rating_histogram": self.rating_histogram,
            "top_raters": [
                {"rater_id": int(node_id), "count": int(count)}
                for node_id, count in self.top_raters
            ],
            "top_profiles": [
                {"profile_id": int(node_id), "count": int(count)}
                for node_id, count in self.top_profiles
            ],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": [
                    "edge_count",
                    "unique_raters",
                    "unique_profiles",
                    "unique_users_union",
                    "overlapping_user_ids",
                    "exclusive_profile_ids",
                    "max_rater_id",
                    "max_profile_id",
                    "mean_rating",
                ],
                "value": [
                    self.edge_count,
                    self.unique_raters,
                    self.unique_profiles,
                    self.unique_users_union,
                    self.overlapping_user_ids,
                    self.exclusive_profile_ids,
                    self.max_rater_id,
                    self.max_profile_id,
                    self.mean_rating,
                ],
            }
        )

    def rating_distribution_frame(self) -> pd.DataFrame:
        rows = [
            {"rating": rating, "count": count}
            for rating, count in sorted(self.rating_histogram.items())
        ]
        frame = pd.DataFrame(rows)
        frame["share"] = frame["count"] / frame["count"].sum()
        return frame


class RecDatingDataset:
    def __init__(self, edges_path: Path | str, chunk_size: int = 1_000_000) -> None:
        self.edges_path = Path(edges_path)
        self.chunk_size = chunk_size

    def iter_chunks(
        self,
        chunk_size: int | None = None,
        nrows: int | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        chunks = pd.read_csv(
            self.edges_path,
            sep=r"\s+",
            names=EDGE_COLUMNS,
            dtype=EDGE_DTYPES,
            chunksize=chunk_size or self.chunk_size,
            nrows=nrows,
        )
        for chunk in chunks:
            if min_rating is not None:
                chunk = chunk[chunk["rating"] >= min_rating]
            if max_rating is not None:
                chunk = chunk[chunk["rating"] <= max_rating]
            if not chunk.empty:
                yield chunk

    def read_edges(
        self,
        nrows: int | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> pd.DataFrame:
        frame = pd.read_csv(
            self.edges_path,
            sep=r"\s+",
            names=EDGE_COLUMNS,
            dtype=EDGE_DTYPES,
            nrows=nrows,
        )
        if min_rating is not None:
            frame = frame[frame["rating"] >= min_rating].copy()
        if max_rating is not None:
            frame = frame[frame["rating"] <= max_rating].copy()
        return frame

    def compute_summary(
        self,
        nrows: int | None = None,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> DatasetSummary:
        raters: set[int] = set()
        profiles: set[int] = set()
        rating_histogram: Counter[int] = Counter()
        rater_counts: defaultdict[int, int] = defaultdict(int)
        profile_counts: defaultdict[int, int] = defaultdict(int)

        edge_count = 0
        rating_sum = 0.0
        max_rater_id = 0
        max_profile_id = 0

        for chunk in self.iter_chunks(nrows=nrows, min_rating=min_rating, max_rating=max_rating):
            raters.update(chunk["rater_id"].unique().tolist())
            profiles.update(chunk["profile_id"].unique().tolist())

            local_rating_hist = chunk["rating"].value_counts().to_dict()
            for rating, count in local_rating_hist.items():
                rating_histogram[int(rating)] += int(count)

            local_raters = chunk["rater_id"].value_counts().to_dict()
            for node_id, count in local_raters.items():
                rater_counts[int(node_id)] += int(count)

            local_profiles = chunk["profile_id"].value_counts().to_dict()
            for node_id, count in local_profiles.items():
                profile_counts[int(node_id)] += int(count)

            edge_count += int(len(chunk))
            rating_sum += float(chunk["rating"].sum())
            max_rater_id = max(max_rater_id, int(chunk["rater_id"].max()))
            max_profile_id = max(max_profile_id, int(chunk["profile_id"].max()))

        overlapping_user_ids = len(raters & profiles)
        unique_users_union = len(raters | profiles)

        return DatasetSummary(
            edge_count=edge_count,
            unique_raters=len(raters),
            unique_profiles=len(profiles),
            unique_users_union=unique_users_union,
            overlapping_user_ids=overlapping_user_ids,
            exclusive_profile_ids=len(profiles - raters),
            max_rater_id=max_rater_id,
            max_profile_id=max_profile_id,
            mean_rating=rating_sum / edge_count if edge_count else 0.0,
            rating_histogram=dict(sorted(rating_histogram.items())),
            top_raters=sorted(rater_counts.items(), key=lambda item: item[1], reverse=True)[:20],
            top_profiles=sorted(profile_counts.items(), key=lambda item: item[1], reverse=True)[:20],
        )
