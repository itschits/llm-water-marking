"""
Abstract base class for all watermarking algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class BaseWatermark(ABC):
    """
    Common interface for watermarking algorithms.

    Subclasses must implement :meth:`generate` and :meth:`get_name`.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
    ) -> str:
        """Return watermarked text generated from *prompt*."""

    # ------------------------------------------------------------------
    @abstractmethod
    def get_name(self) -> str:
        """Human-readable algorithm name."""

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Multinomial sampling with temperature scaling.

        Parameters
        ----------
        logits : shape (vocab_size,)
        temperature : float  > 0; set to a very small value for near-greedy.

        Returns
        -------
        Scalar tensor containing the sampled token id.
        """
        temperature = max(temperature, 1e-6)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(0)
