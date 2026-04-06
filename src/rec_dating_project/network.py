from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse

from .dataset import DatasetSummary, RecDatingDataset


@dataclass
class BipartiteSnapshot:
    matrix: sparse.csr_matrix
    edge_count: int
    num_raters: int
    num_profiles: int

    @property
    def density(self) -> float:
        total = self.num_raters * self.num_profiles
        return float(self.edge_count / total) if total else 0.0

    def __repr__(self) -> str:
        return (
            "BipartiteSnapshot("
            f"shape={self.matrix.shape}, "
            f"edge_count={self.edge_count}, "
            f"density={self.density:.6f})"
        )


class RoleBasedBipartiteNetwork:
    def __init__(self, dataset: RecDatingDataset) -> None:
        self.dataset = dataset

    def build_sparse_rating_matrix(
        self,
        nrows: int | None = None,
        summary: DatasetSummary | None = None,
        dtype: np.dtype = np.float64,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> BipartiteSnapshot:
        local_summary = summary or self.dataset.compute_summary(
            nrows=nrows,
            min_rating=min_rating,
            max_rating=max_rating,
        )

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []

        for chunk in self.dataset.iter_chunks(
            nrows=nrows,
            min_rating=min_rating,
            max_rating=max_rating,
        ):
            row_parts.append(chunk["rater_id"].to_numpy(dtype=np.int32, copy=False) - 1)
            col_parts.append(chunk["profile_id"].to_numpy(dtype=np.int32, copy=False) - 1)
            data_parts.append(chunk["rating"].to_numpy(dtype=dtype, copy=False))

        if not row_parts:
            raise ValueError("No edges were loaded from the dataset.")

        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        data = np.concatenate(data_parts)

        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(local_summary.max_rater_id, local_summary.max_profile_id),
            dtype=dtype,
        )
        matrix.sum_duplicates()

        return BipartiteSnapshot(
            matrix=matrix,
            edge_count=int(matrix.nnz),
            num_raters=matrix.shape[0],
            num_profiles=matrix.shape[1],
        )

    def build_networkx_sample(
        self,
        nrows: int = 50_000,
        min_rating: int | None = None,
        max_rating: int | None = None,
    ) -> nx.DiGraph:
        edges = self.dataset.read_edges(nrows=nrows, min_rating=min_rating, max_rating=max_rating)

        graph = nx.DiGraph()
        for row in edges.itertuples(index=False):
            rater_node = f"rater::{int(row.rater_id)}"
            profile_node = f"profile::{int(row.profile_id)}"

            graph.add_node(rater_node, role="rater", raw_id=int(row.rater_id), bipartite=0)
            graph.add_node(profile_node, role="profile", raw_id=int(row.profile_id), bipartite=1)
            graph.add_edge(rater_node, profile_node, weight=int(row.rating))

        return graph
