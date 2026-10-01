"""
Abstract base class shared by all watermark detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch


class BaseDetector(ABC):
    """
    Common interface for watermark detection.

    Every concrete detector must implement three methods:
    :meth:`score`, :meth:`detect`, and :meth:`get_name`.
    """

    @abstractmethod
    def score(self, token_ids: Union[torch.Tensor, list]) -> float:
        """
        Return a scalar detection score for *token_ids*.

        Higher values indicate higher confidence of a watermark.
        The interpretation (z-score, mean g-value, …) depends on the
        concrete subclass.
        """

    @abstractmethod
    def detect(self, token_ids: Union[torch.Tensor, list]) -> Tuple[bool, float]:
        """
        Decide whether *token_ids* contains a watermark.

        Returns
        -------
        (is_watermarked, score)
        """

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable detector name."""

    # ------------------------------------------------------------------
    @staticmethod
    def _to_list(token_ids: Union[torch.Tensor, list]) -> list:
        if isinstance(token_ids, torch.Tensor):
            return token_ids.flatten().tolist()
        return list(token_ids)
