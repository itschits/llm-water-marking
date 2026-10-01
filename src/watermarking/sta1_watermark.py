"""
STA-1  —  Sampling One Then Accepting

Unlike KGW (Kirchenbauer et al., 2023) which adds a logit bias δ to
green-list tokens and thereby directly distorts the model distribution,
STA-1 **never modifies logits**.  Instead it uses rejection sampling:

Algorithm (generation)
----------------------
At each step t:

1. Derive a deterministic green list G(x_{t-1}) by hashing the previous
   token id together with a global *seed* (SHA-256 → NumPy shuffle).
   |G| = ⌊γ · |V|⌋ tokens.

2. Sample a candidate token c from the **unmodified** distribution
   p(·|x_{<t})  (temperature softmax + multinomial).

3. If c ∈ G  →  accept and emit c.
   Else       →  reject c and draw again  (up to *max_attempts* retries).

4. Fallback: if no green token is drawn within *max_attempts*, accept the
   last candidate to avoid degrading fluency.

Because the underlying distribution is never changed, STA-1 preserves
text quality even in low-entropy situations where one token strongly
dominates (e.g. P("Paris") = 0.95).  KGW would redistribute that
probability mass; STA-1 leaves the distribution untouched.

Detection uses the same one-sided z-test as KGW (see :class:`STA1Detector`).
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base_watermark import BaseWatermark


class STA1Watermark(BaseWatermark):
    """
    STA-1 — Sampling One Then Accepting.

    Uses rejection sampling to embed a watermark without modifying the
    model's logit distribution.

    Parameters
    ----------
    vocab_size : int
    seed : int
        Global seed mixed with the previous token id when hashing.
    gamma : float
        Fraction of the vocabulary placed in the green list  (0 < γ < 1).
    delta : float
        Unused in STA-1 generation (kept for config/API compatibility).
        In KGW this is the logit bias; STA-1 never modifies logits.
    max_attempts : int
        Maximum rejection-sampling retries per token before falling back
        to the last drawn candidate.  Higher values strengthen the
        watermark signal at a minor generation-time cost.
    """

    def __init__(
        self,
        vocab_size: int,
        seed: int = 42,
        gamma: float = 0.5,
        delta: float = 2.0,
        max_attempts: int = 20,
    ) -> None:
        super().__init__(vocab_size)
        self.seed = seed
        self.gamma = gamma
        self.delta = delta  # kept for config compatibility; unused in generation
        self.max_attempts = max_attempts
        self._greenlist_size = max(1, int(gamma * vocab_size))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_greenlist(self, prev_token_id: int) -> torch.Tensor:
        """
        Return the green-list token ids for a given *prev_token_id*.

        The list is fully determined by (seed, prev_token_id) and is
        computed via SHA-256 so that results are identical across
        platforms and Python versions.
        """
        hash_input = f"{self.seed}:{prev_token_id}".encode()
        hash_bytes = hashlib.sha256(hash_input).digest()
        rng_seed = int.from_bytes(hash_bytes[:4], "big")

        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(self.vocab_size)
        # Convert to plain Python list first to avoid torch/NumPy 2.x bridge issue
        green_ids = perm[: self._greenlist_size].tolist()
        return torch.tensor(green_ids, dtype=torch.long)

    def _get_greenlist_set(self, prev_token_id: int) -> frozenset:
        """
        Return the green list as a frozenset for O(1) membership testing
        (used by the detector).
        """
        hash_input = f"{self.seed}:{prev_token_id}".encode()
        hash_bytes = hashlib.sha256(hash_input).digest()
        rng_seed = int.from_bytes(hash_bytes[:4], "big")

        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(self.vocab_size)
        return frozenset(int(x) for x in perm[: self._greenlist_size])

    def _rejection_sample(
        self, logits: torch.Tensor, green_list: frozenset, temperature: float
    ) -> torch.Tensor:
        """
        Draw from the **unmodified** distribution and re-draw up to
        *max_attempts* times until a green-list token is found.

        Falls back to the last drawn token if the budget is exhausted,
        preserving text fluency.
        """
        token = self._sample_token(logits, temperature)
        for _ in range(self.max_attempts - 1):
            if token.item() in green_list:
                return token
            token = self._sample_token(logits, temperature)
        return token  # accept whatever was last drawn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_greenlist_for_token(self, prev_token_id: int) -> frozenset:
        """Expose green list as frozenset for the detector (O(1) lookup)."""
        return self._get_greenlist_set(prev_token_id)

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate watermarked text using STA-1 rejection sampling at every step.

        Parameters
        ----------
        model, tokenizer : loaded HuggingFace objects
        prompt : str   — prefix fed to the model
        max_new_tokens : int
        temperature : float

        Returns
        -------
        str  — newly generated tokens decoded to text (prompt NOT included).
        """
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(generated_ids)
                logits = outputs.logits[0, -1, :].float()

                prev_token = generated_ids[0, -1].item()
                green_list = self._get_greenlist_set(prev_token)

                next_token = self._rejection_sample(logits, green_list, temperature)
                generated_ids = torch.cat(
                    [generated_ids, next_token.view(1, 1)], dim=1
                )

                if next_token.item() == tokenizer.eos_token_id:
                    break

        new_ids = generated_ids[0, prompt_len:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    def get_name(self) -> str:
        return "STA-1"
