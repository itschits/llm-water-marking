"""
Model loader for causal language models via HuggingFace Transformers.

Supports any AutoModelForCausalLM-compatible checkpoint.  Short aliases
for commonly used lightweight models are provided in SUPPORTED_MODELS.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Short-name → HuggingFace hub id
SUPPORTED_MODELS: dict[str, str] = {
    "gpt2":          "openai-community/gpt2",
    "gpt2-medium":   "openai-community/gpt2-medium",
    "gpt2-large":    "openai-community/gpt2-large",
    "tinyllama":     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "opt-125m":      "facebook/opt-125m",
    "opt-1.3b":      "facebook/opt-1.3b",
    "opt-2.7b":      "facebook/opt-2.7b",
    "phi-2":         "microsoft/phi-2",
    "gemma-2b":      "google/gemma-2b",
    "mistral-7b":    "mistralai/Mistral-7B-v0.1",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-2.5-3b":  "Qwen/Qwen2.5-3B",
}


class ModelLoader:
    """
    Loads a causal LM and its tokenizer.

    Parameters
    ----------
    model_name : str
        Either a short alias from SUPPORTED_MODELS or a full HuggingFace id.
    device : str, optional
        'cuda', 'cpu', or 'auto'.  Defaults to CUDA when available.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    # ------------------------------------------------------------------
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Download / cache and return (model, tokenizer)."""
        hub_id = SUPPORTED_MODELS.get(self.model_name, self.model_name)
        logger.info("Loading '%s' on device='%s'", hub_id, self.device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hub_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model – use float16 on CUDA to reduce memory footprint
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            hub_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        vocab = self.get_vocab_size()
        logger.info("Loaded '%s' — vocab size: %d", hub_id, vocab)
        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    def get_vocab_size(self) -> int:
        if self.model is None:
            raise RuntimeError("Call load() before get_vocab_size().")
        return self.model.config.vocab_size

    # ------------------------------------------------------------------
    def get_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Call load() before get_device().")
        return next(self.model.parameters()).device
