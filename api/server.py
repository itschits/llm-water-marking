"""
FastAPI backend — LLM Watermarking API
=======================================

Endpoints
---------
GET  /api/health           Health check
POST /api/detect           Detect watermark in submitted text
GET  /api/results          Return latest experiment results from results/
GET  /api/config           Return active watermarking configuration
"""

from __future__ import annotations

import glob
import gc
import json
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Literal, Optional, Sequence

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from scipy import stats

# ── Project root on path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detection.sta1_detector import STA1Detector
from src.detection.synthid_detector import SynthIDDetector
from src.watermarking.sta1_watermark import STA1Watermark
from src.watermarking.synthid_watermark import SynthIDWatermark
from src.models.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
_cfg_path = os.path.join(ROOT, "config.yaml")
with open(_cfg_path, encoding="utf-8") as _fh:
    CFG: Dict[str, Any] = yaml.safe_load(_fh)

# ── Watermark / detector instances (no full model needed for detection)
VOCAB_SIZE = 50257  # GPT-2 / compatible vocab; override via config if needed

_sta1_cfg = CFG["watermarking"]["sta1"]
_sid_cfg = CFG["watermarking"]["synthid"]

STA1 = STA1Watermark(VOCAB_SIZE, **_sta1_cfg)
SYNTHID = SynthIDWatermark(VOCAB_SIZE, **_sid_cfg)
STA1_DET = STA1Detector(STA1, z_threshold=CFG["detection"]["sta1"]["z_threshold"])
SYNTHID_DET = SynthIDDetector(SYNTHID, z_threshold=CFG["detection"]["synthid"]["z_threshold"])

# Lazy-load GPT-2 tokenizer only when first /detect call arrives
_tokenizer = None
_experiment_lock = threading.Lock()

RUNNABLE_MODELS = {
    "gpt2": {
        "label": "GPT-2 (124M)",
        "hub_id": "openai-community/gpt2",
        "warning": None,
    },
    "qwen-2.5-3b": {
        "label": "Qwen 2.5 3B",
        "hub_id": "Qwen/Qwen2.5-3B",
        "warning": "This 3B model needs substantially more RAM/VRAM and may take longer to load.",
    },
    "qwen-2.5-1.5b": {
        "label": "Qwen 2.5 1.5B",
        "hub_id": "Qwen/Qwen2.5-1.5B",
        "warning": "The first run downloads the model; generation speed depends on your hardware.",
    },
}

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        logger.info("Loading GPT-2 tokenizer …")
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _tokenizer


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Watermarking API",
    description="Detection and results API for STA-1 and SynthID-Text watermarks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class AlgorithmResult(BaseModel):
    z_score: float
    detected: bool
    threshold: float


class DetectResponse(BaseModel):
    token_count: int
    sta1: AlgorithmResult
    synthid: AlgorithmResult


class RunExperimentRequest(BaseModel):
    """One interactive, plain-vs-watermarked generation run."""

    model: Literal["gpt2", "qwen-2.5-1.5b", "qwen-2.5-3b"] = "qwen-2.5-1.5b"
    algorithm: Literal["sta1", "synthid"] = "sta1"
    prompt: str = "Artificial intelligence is transforming the world because"
    max_new_tokens: int = Field(default=120, ge=10, le=256)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def prompt_is_usable(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        if len(value) > 2_000:
            raise ValueError("prompt must be 2,000 characters or fewer")
        return value


class GeneratedTextResult(BaseModel):
    text: str
    token_count: int
    z_score: float
    p_value: float
    detected: bool
    signal_label: str
    signal_value: float
    signal_detail: str


class RunExperimentResponse(BaseModel):
    model: str
    model_id: str
    algorithm: str
    prompt: str
    threshold: float
    elapsed_seconds: float
    plain: GeneratedTextResult
    watermarked: GeneratedTextResult


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/detect", response_model=DetectResponse)
def detect(body: DetectRequest) -> DetectResponse:
    """
    Tokenise *text* with the GPT-2 tokenizer and run both detectors.
    Returns z-scores and binary detection decisions.
    """
    tok = _get_tokenizer()
    ids = tok.encode(body.text)

    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Text too short (need ≥ 2 tokens).")

    sta1_z = STA1_DET.score(ids)
    sta1_detected, _ = STA1_DET.detect(ids)
    sid_z = SYNTHID_DET.score(ids)
    sid_detected, _ = SYNTHID_DET.detect(ids)

    return DetectResponse(
        token_count=len(ids),
        sta1=AlgorithmResult(
            z_score=round(sta1_z, 3),
            detected=sta1_detected,
            threshold=STA1_DET.z_threshold,
        ),
        synthid=AlgorithmResult(
            z_score=round(sid_z, 3),
            detected=sid_detected,
            threshold=SYNTHID_DET.z_threshold,
        ),
    )


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(0)


def _generate_token_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    choose_token: Callable[[torch.Tensor, Sequence[int]], torch.Tensor],
) -> tuple[list[int], list[int]]:
    """Generate with a KV cache and retain the original prompt token context."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    full_ids = input_ids[0].tolist()
    continuation: list[int] = []

    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        logits = output.logits[0, -1].float()
        cache = output.past_key_values
        for _ in range(max_new_tokens):
            token = choose_token(logits, full_ids)
            token_id = int(token.item())
            continuation.append(token_id)
            full_ids.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            output = model(
                input_ids=token.view(1, 1).to(device),
                past_key_values=cache,
                use_cache=True,
            )
            logits = output.logits[0, -1].float()
            cache = output.past_key_values
    return full_ids, continuation


def _score_sta1(
    watermark: STA1Watermark, full_ids: list[int], continuation_start: int
) -> tuple[float, float, int, str]:
    total = len(full_ids) - continuation_start
    if total <= 0:
        return 0.0, 1.0, 0, "0 / 0 green tokens"
    green = sum(
        full_ids[index] in watermark.get_greenlist_for_token(full_ids[index - 1])
        for index in range(continuation_start, len(full_ids))
    )
    expected = watermark.gamma * total
    std = math.sqrt(total * watermark.gamma * (1.0 - watermark.gamma))
    z_score = (green - expected) / std if std else 0.0
    return z_score, float(stats.norm.sf(z_score)), green, f"{green} / {total} green tokens"


def _score_synthid(
    watermark: SynthIDWatermark, full_ids: list[int], continuation_start: int
) -> tuple[float, float, float, str]:
    scores = [
        watermark.score_token(full_ids[index], full_ids[max(0, index - watermark.depth):index])
        for index in range(continuation_start, len(full_ids))
    ]
    if not scores:
        return 0.0, 1.0, 0.5, "mean g-value: 0.5000 (0 tokens)"
    mean_g = float(np.mean(scores))
    z_score = math.sqrt(12.0 * len(scores)) * (mean_g - 0.5)
    return z_score, float(stats.norm.sf(z_score)), mean_g, f"mean g-value: {mean_g:.4f} ({len(scores)} tokens)"


@app.post("/api/experiments/run", response_model=RunExperimentResponse)
def run_experiment(body: RunExperimentRequest) -> RunExperimentResponse:
    """Run a selected model and watermark algorithm, returning browser-ready data."""
    if not _experiment_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An experiment is already running. Please wait for it to finish.")

    started = time.perf_counter()
    model = None
    try:
        model_info = RUNNABLE_MODELS[body.model]
        logger.info("Starting interactive %s experiment with %s", body.algorithm, model_info["hub_id"])
        loader = ModelLoader(body.model)
        model, tokenizer = loader.load()
        vocab_size = loader.get_vocab_size()

        if body.algorithm == "sta1":
            watermark = STA1Watermark(vocab_size, **CFG["watermarking"]["sta1"])
            threshold = CFG["detection"]["sta1"]["z_threshold"]

            def choose_watermarked(logits: torch.Tensor, context: Sequence[int]) -> torch.Tensor:
                green_list = watermark.get_greenlist_for_token(context[-1])
                return watermark._rejection_sample(logits, green_list, body.temperature)

            def score(ids: list[int], start: int):
                z, p, signal, detail = _score_sta1(watermark, ids, start)
                return z, p, signal, detail, "Green tokens"

        else:
            watermark = SynthIDWatermark(vocab_size, **CFG["watermarking"]["synthid"])
            threshold = CFG["detection"]["synthid"]["z_threshold"]

            def choose_watermarked(logits: torch.Tensor, context: Sequence[int]) -> torch.Tensor:
                return watermark._tournament_sample(logits, list(context[-watermark.depth:]), body.temperature)

            def score(ids: list[int], start: int):
                z, p, signal, detail = _score_synthid(watermark, ids, start)
                return z, p, signal, detail, "Mean g-value"

        plain_full, plain_ids = _generate_token_ids(
            model, tokenizer, body.prompt, body.max_new_tokens,
            lambda logits, _: _sample_token(logits, body.temperature),
        )
        watermarked_full, watermarked_ids = _generate_token_ids(
            model, tokenizer, body.prompt, body.max_new_tokens, choose_watermarked,
        )

        def make_result(full_ids: list[int], continuation: list[int]) -> GeneratedTextResult:
            start = len(full_ids) - len(continuation)
            z_score, p_value, signal, detail, signal_label = score(full_ids, start)
            return GeneratedTextResult(
                text=tokenizer.decode(continuation, skip_special_tokens=True),
                token_count=len(continuation),
                z_score=round(z_score, 4),
                p_value=p_value,
                detected=z_score >= threshold,
                signal_label=signal_label,
                signal_value=round(signal, 4),
                signal_detail=detail,
            )

        return RunExperimentResponse(
            model=model_info["label"],
            model_id=model_info["hub_id"],
            algorithm="STA-1" if body.algorithm == "sta1" else "SynthID-Text",
            prompt=body.prompt,
            threshold=threshold,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            plain=make_result(plain_full, plain_ids),
            watermarked=make_result(watermarked_full, watermarked_ids),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Interactive experiment failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _experiment_lock.release()


@app.get("/api/results")
def get_results() -> Dict[str, Any]:
    """Return the most recent experiment results JSON from results/."""
    results_dir = os.path.join(ROOT, "results")
    files = sorted(glob.glob(os.path.join(results_dir, "results_*.json")))
    if not files:
        return {"available": False, "data": None, "filename": None}

    latest = files[-1]
    with open(latest, encoding="utf-8") as fh:
        data = json.load(fh)

    return {
        "available": True,
        "filename": os.path.basename(latest),
        "data": data,
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    """Return current watermarking configuration (sans private keys)."""
    return {
        "sta1": CFG["watermarking"]["sta1"],
        "synthid": {k: v for k, v in CFG["watermarking"]["synthid"].items() if k != "keys"},
        "detection": CFG["detection"],
        "generation": CFG["generation"],
    }
