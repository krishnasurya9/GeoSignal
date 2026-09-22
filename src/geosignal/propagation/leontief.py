"""
geosignal.propagation.leontief
Leontief Input-Output shock propagation.

Models how a supply/demand shock in one sector propagates through
inter-industry linkages via the Leontief inverse: L = (I - A)^{-1}

Current status (L-006, DEC-009):
  The A matrix used in experiments is a 5-sector illustrative toy matrix.
  All outputs must be labelled ILLUSTRATIVE until replaced with real
  India MoSPI 2017-18 or WIOD data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class LeontiefPropagator:
    """
    Computes total output impact of a sector shock using the Leontief inverse.

    Args:
        io_matrix:     Square technical coefficient matrix A (n×n).
                       A[i,j] = value of input from sector i per unit of sector j.
        sector_labels: Ordered list of sector names (length n).
    """

    def __init__(self, io_matrix: np.ndarray, sector_labels: list[str]):
        n = io_matrix.shape[0]
        if io_matrix.shape != (n, n):
            raise ValueError("io_matrix must be square.")
        if len(sector_labels) != n:
            raise ValueError("sector_labels length must match matrix size.")

        self.A             = io_matrix
        self.sector_labels = sector_labels
        self.n             = n
        self.L             = np.linalg.inv(np.eye(n) - io_matrix)
        logger.info(f"Leontief inverse computed for {n} sectors.")

    def propagate_shock(self, shock_sector: str,
                        shock_magnitude: float) -> pd.Series:
        """
        Compute total output change from a demand/supply shock in one sector.

        Args:
            shock_sector:    Name of the shocked sector (must be in sector_labels).
            shock_magnitude: Shock size, e.g. -0.15 = 15% reduction.

        Returns:
            pd.Series of Δ total output per sector, sorted ascending.
        """
        if shock_sector not in self.sector_labels:
            raise ValueError(f"'{shock_sector}' not in sector_labels.")

        idx       = self.sector_labels.index(shock_sector)
        delta_d   = np.zeros(self.n)
        delta_d[idx] = shock_magnitude
        delta_x   = self.L @ delta_d

        return pd.Series(delta_x, index=self.sector_labels).sort_values()

    @classmethod
    def from_csv(cls, path: Path) -> "LeontiefPropagator":
        """
        Load A matrix from a CSV (first row and column = sector labels).
        Use this to plug in real India MoSPI / WIOD I-O tables.
        """
        df = pd.read_csv(path, index_col=0)
        return cls(io_matrix=df.values, sector_labels=list(df.index))

    @classmethod
    def toy_5sector(cls) -> "LeontiefPropagator":
        """
        Return the illustrative 5-sector toy propagator used in experiments.
        ILLUSTRATIVE ONLY — do not interpret as empirical results (L-006).
        """
        labels = ["Energy", "Semiconductor", "Automotive", "Logistics", "ConsumerGoods"]
        A = np.array([
            [0.10, 0.05, 0.08, 0.15, 0.04],
            [0.02, 0.12, 0.20, 0.01, 0.05],
            [0.03, 0.08, 0.10, 0.10, 0.12],
            [0.15, 0.04, 0.07, 0.05, 0.08],
            [0.01, 0.02, 0.03, 0.04, 0.10],
        ])
        return cls(A, labels)
