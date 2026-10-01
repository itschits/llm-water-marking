"""
SynthID-Text  —  Tournament-sampling watermark

Reference
---------
Dathathri et al., "Scalable watermarking for identifying large language model
outputs", Nature 2024 / Google DeepMind.
https://www.nature.com/articles/s41586-024-08025-4

Algorithm  (tournament sampling variant)
----------------------------------------
A set of D secret keys k_0 … k_{D-1} is shared between the generator and
the detector.

Generation (at position t):

1. Sample B *candidate* tokens  c_1 … c_B  i.i.d. from the base distribution
   p(·|x_{<t}).
2. For each candidate c, compute a pseudo-random score

       score(c) = (1/D) Σ_{d=0}^{D-1}  g(k_d, x_{t-d-1}, c)   ∈ [0, 1]

   where  g(·)  hashes its three arguments to a float in [0, 1] using SHA-256.
3. Emit the candidate with the *highest* score.

Under this scheme, watermarked tokens have scores skewed above 0.5, while
tokens drawn from an unmodified distribution have scores near 0.5.

Detection uses the same  g(·)  function to compute a sequence-level z-score
(see :class:`SynthIDDetector`).
"""

from __future__ import annotations

import hashlib
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base_watermark import BaseWatermark


class SynthIDWatermark(BaseWatermark):
    """
    SynthID-Text watermarking via tournament sampling.

    Parameters
    ----------
    vocab_size : int
    keys : list[int]
        Secret integer keys — one per hash depth level.
        The number of keys also determines *depth*.
    depth : int
        Number of preceding context tokens considered in the hash.
        Overridden by ``len(keys)`` when keys are supplied.
    n_candidates : int
        Tournament size B — number of candidates sampled from the base
        distribution before selecting the highest-scoring one.
    """

    def __init__(
        self,
        vocab_size: int,
        keys: List[int] = None,
        depth: int = 4,
        n_candidates: int = 5,
    ) -> None:
        super().__init__(vocab_size)
        self.keys = keys if keys is not None else [2024, 42, 1234, 5678]
        self.depth = len(self.keys)  # depth = number of keys
        self.n_candidates = n_candidates

    # ------------------------------------------------------------------
    # Core hash primitive
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_to_float(key: int, context_token: int, candidate: int) -> float:
        """
        Map (key, context_token, candidate) → float in [0, 1].

        Uses the first 4 bytes of SHA-256 to give a well-distributed value
        that is fully reproducible across platforms and Python versions.
        """
        raw = f"{key}:{context_token}:{candidate}".encode()
        digest = hashlib.sha256(raw).digest()
        int_val = int.from_bytes(digest[:4], "big")
        return int_val / (2**32 - 1)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_token(self, token_id: int, context_tokens: List[int]) -> float:
        """
        Compute the watermark score for *token_id* given *context_tokens*.

        Returns a value in [0, 1]; values > 0.5 indicate the token is
        aligned with the watermark signal.

        Parameters
        ----------
        token_id : int
        context_tokens : list[int]
            Preceding token ids (most recent is last).  At least *depth*
            tokens are used; shorter lists are zero-padded.
        """
        total = 0.0
        for d, key in enumerate(self.keys):
            # x_{t - d - 1}: index backwards into the context
            idx = -(d + 1)
            ctx_tok = context_tokens[idx] if d < len(context_tokens) else 0
            total += self._hash_to_float(key, ctx_tok, token_id)
        return total / self.depth

    # ------------------------------------------------------------------
    # Tournament sampling
    # ------------------------------------------------------------------

    def _tournament_sample(
        self,
        logits: torch.Tensor,
        context_tokens: List[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample *n_candidates* from the base distribution and return the
        one with the highest watermark score.
        """
        temperature = max(temperature, 1e-6)
        probs = torch.softmax(logits / temperature, dim=-1)

        n = min(self.n_candidates, self.vocab_size)
        candidates = torch.multinomial(probs, n, replacement=False)

        scores = [self.score_token(c.item(), context_tokens) for c in candidates]
        best = max(range(n), key=lambda i: scores[i])
        return candidates[best]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate watermarked text using tournament sampling at every step.

        Returns
        -------
        str  — newly generated text (prompt NOT included).
        """
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(generated_ids)
                logits = outputs.logits[0, -1, :].float()

                # Extract context window for hashing
                ctx = generated_ids[0, -self.depth:].tolist()

                next_token = self._tournament_sample(logits, ctx, temperature)
                generated_ids = torch.cat(
                    [generated_ids, next_token.view(1, 1)], dim=1
                )

                if next_token.item() == tokenizer.eos_token_id:
                    break

        new_ids = generated_ids[0, prompt_len:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    def get_name(self) -> str:
        return "SynthID-Text"
