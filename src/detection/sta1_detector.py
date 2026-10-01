"""
STA-1 Watermark Detector

Implements the one-sided z-test described in:
    Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.

Detection procedure
-------------------
For each consecutive token pair (x_{t-1}, x_t) in the candidate text:

  1. Reconstruct the green list G(x_{t-1}) deterministically from the
     watermark parameters.
  2. Increment a green-token counter if  x_t ∈ G(x_{t-1}).

Let s = green-token count,  T = total pairs,  γ = green-list fraction.

Under H₀ (no watermark):  E[s] = γT,  Var[s] = Tγ(1-γ)

z = (s − γT) / √(Tγ(1−γ))

Reject H₀ (i.e. declare watermark present) when z ≥ z_threshold.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
from scipy import stats

from .base_detector import BaseDetector
from ..watermarking.sta1_watermark import STA1Watermark


class STA1Detector(BaseDetector):
    """
    Detects the STA-1 green/red-list watermark.

    Parameters
    ----------
    watermark : STA1Watermark
        Must have the same *seed* and *gamma* as used during generation.
    z_threshold : float
        One-sided z-test threshold.  Default 4.0 gives FPR ≈ 3 × 10⁻⁵.
    """

    def __init__(self, watermark: STA1Watermark, z_threshold: float = 4.0) -> None:
        self.watermark = watermark
        self.z_threshold = z_threshold

    # ------------------------------------------------------------------
    def _count_green_tokens(self, ids: list) -> Tuple[int, int]:
        """Return (green_count, total_pairs)."""
        green_count = 0
        total = len(ids) - 1  # need at least one preceding token
        if total <= 0:
            return 0, 0

        for i in range(1, len(ids)):
            gl = self.watermark.get_greenlist_for_token(ids[i - 1])
            if ids[i] in gl:  # frozenset → O(1) lookup
                green_count += 1

        return green_count, total

    # ------------------------------------------------------------------
    def score(self, token_ids: Union[torch.Tensor, list]) -> float:
        """
        Compute the z-score for *token_ids*.

        A z-score above :attr:`z_threshold` indicates the text is
        (likely) watermarked.
        """
        ids = self._to_list(token_ids)
        green_count, total = self._count_green_tokens(ids)

        if total == 0:
            return 0.0

        gamma = self.watermark.gamma
        expected = gamma * total
        std = math.sqrt(total * gamma * (1.0 - gamma))
        if std == 0.0:
            return 0.0

        return (green_count - expected) / std

    # ------------------------------------------------------------------
    def detect(self, token_ids: Union[torch.Tensor, list]) -> Tuple[bool, float]:
        """
        Returns
        -------
        (is_watermarked, z_score)
        """
        z = self.score(token_ids)
        return z >= self.z_threshold, z

    # ------------------------------------------------------------------
    def get_p_value(self, token_ids: Union[torch.Tensor, list]) -> float:
        """One-tailed p-value for the z-test."""
        return float(1.0 - stats.norm.cdf(self.score(token_ids)))

    def get_name(self) -> str:
        return "STA-1 Detector"
