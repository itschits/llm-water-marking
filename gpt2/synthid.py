"""
GPT-2  ×  SynthID-Text  (Tournament Sampling)
===============================================
Completely self-contained — no dependency on src/.

Algorithm
---------
SynthID-Text embeds a watermark by changing WHICH token is selected,
not how probabilities are assigned:

  Shared secret: KEYS = [k₀, k₁, k₂, k₃]   (one key per depth level)

  At each step t:
  1. Sample N_CANDIDATES tokens i.i.d. from the UNMODIFIED distribution.
  2. For each candidate c, compute a watermark score in [0, 1]:

       score(c) = (1/D) Σ_{d=0}^{D-1}  hash(k_d, x_{t-d-1}, c)

     where hash(·) → SHA-256 → first 4 bytes as uint32 / (2³²−1).

  3. Emit the candidate with the HIGHEST score.

  Because the winner always beats its competitors, its score is biased
  above 0.5 across a long sequence — this is the detectable signal.

Detection computes per-token g-values and applies a one-sided z-test:
  z = √(12N) · (mean_g − 0.5)

Run
---
    python experiments/gpt2/synthid.py
"""

import hashlib
import math
import textwrap
from typing import List

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────
MODEL_ID       = "openai-community/gpt2"
PROMPT         = "Artificial intelligence is transforming the world because"
MAX_NEW_TOKENS = 120
TEMPERATURE    = 1.0

# SynthID watermark parameters
KEYS           = [2024, 42, 1234, 5678]  # secret keys — one per depth level
DEPTH          = len(KEYS)               # context window used in hashing
N_CANDIDATES   = 5                       # tournament size (larger = stronger signal)
Z_THRESHOLD    = 4.0                     # detection threshold (FPR ≈ 3e-5)


# ── Hash primitive ─────────────────────────────────────────────────────

def hash_to_float(key: int, context_token: int, candidate: int) -> float:
    """
    Map (key, context_token, candidate) → float in [0, 1].
    SHA-256 ensures uniform distribution; fully reproducible.
    """
    raw    = f"{key}:{context_token}:{candidate}".encode()
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "big") / (2**32 - 1)


# ── Watermark scoring ──────────────────────────────────────────────────

def score_token(token_id: int, context_tokens: List[int]) -> float:
    """
    Compute the watermark score for a token given its preceding context.

    Averages D hash values, one per key/depth level.
    Values > 0.5 indicate alignment with the watermark signal.
    """
    total = 0.0
    for d, key in enumerate(KEYS):
        ctx_tok = context_tokens[-(d + 1)] if d < len(context_tokens) else 0
        total  += hash_to_float(key, ctx_tok, token_id)
    return total / DEPTH


# ── Tournament sampling ────────────────────────────────────────────────

def tournament_sample(logits: torch.Tensor, context_tokens: List[int],
                      temperature: float = 1.0) -> torch.Tensor:
    """
    Draw N_CANDIDATES tokens from the unmodified distribution;
    return the one with the highest watermark score.
    Logits are NEVER modified.
    """
    probs      = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    n          = min(N_CANDIDATES, logits.shape[0])
    candidates = torch.multinomial(probs, n, replacement=False)
    scores     = [score_token(c.item(), context_tokens) for c in candidates]
    best       = max(range(n), key=lambda i: scores[i])
    return candidates[best]


# ── Model loading ──────────────────────────────────────────────────────

def load_model():
    print(f"\n  Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Vocab size: {model.config.vocab_size:,}")
    return model, tokenizer


# ── Generation ─────────────────────────────────────────────────────────

def generate_plain(model, tokenizer, prompt: str) -> str:
    """Standard sampling — no watermark applied."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


def generate_watermarked(model, tokenizer, prompt: str) -> str:
    """SynthID watermarked generation — manual autoregressive loop."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]
    generated  = input_ids.clone()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(generated).logits[0, -1, :].float()   # unmodified
            ctx    = generated[0, -DEPTH:].tolist()               # context window
            tok    = tournament_sample(logits, ctx, TEMPERATURE)
            generated = torch.cat([generated, tok.view(1, 1)], dim=1)
            if tok.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)


# ── Detection ──────────────────────────────────────────────────────────

def detect(token_ids: list):
    """
    Compute per-token g-values, then a z-score against the null (Uniform).

    H₀: each g_t ~ Uniform(0,1)  →  mean_g ~ N(0.5, 1/(12N))
    z  = √(12N) · (mean_g − 0.5)
    """
    g_vals = [
        score_token(token_ids[t], token_ids[max(0, t - DEPTH):t])
        for t in range(1, len(token_ids))
    ]

    n = len(g_vals)
    if n == 0:
        return 0.0, 0.5, 0, 1.0

    mean_g   = float(np.mean(g_vals))
    std_null = math.sqrt(1.0 / (12.0 * n))
    z        = (mean_g - 0.5) / std_null if std_null > 0 else 0.0
    p_value  = float(1.0 - stats.norm.cdf(z))
    return z, mean_g, n, p_value


# ── Display helpers ────────────────────────────────────────────────────

def _bar(z: float, width: int = 32) -> str:
    scale  = Z_THRESHOLD * 2
    filled = int(round(max(0.0, min(z, scale)) / scale * width))
    mid    = width // 2
    bar    = list("█" * filled + "░" * (width - filled))
    if mid < width:
        bar[mid] = "|"
    return "".join(bar)


def _wrap(text: str) -> str:
    snippet = text[:300] + ("…" if len(text) > 300 else "")
    return textwrap.fill(snippet, width=72,
                         initial_indent="    ", subsequent_indent="    ")


def print_result(label: str, text: str, z: float,
                 mean_g: float, n: int, p_val: float) -> None:
    detected = z >= Z_THRESHOLD
    verdict  = "WATERMARKED ✓" if detected else "NOT detected ✗"
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    print(_wrap(text))
    print()
    print(f"  Mean g-value : {mean_g:.4f}  (null = 0.5000,  N = {n} tokens)")
    print(f"  Z-score      : {z:.4f}   {_bar(z)}")
    print(f"  P-value      : {p_val:.2e}")
    print(f"  Decision     : z ≥ {Z_THRESHOLD} →  {verdict}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  GPT-2  ×  SynthID-Text  (Tournament Sampling)")
    print("=" * 72)
    print(f"\n  Model        : {MODEL_ID}")
    print(f"  Prompt       : {PROMPT!r}")
    print(f"  Max tok      : {MAX_NEW_TOKENS}   temperature: {TEMPERATURE}")
    print(f"  Keys         : {KEYS}")
    print(f"  Depth        : {DEPTH}   n_candidates: {N_CANDIDATES}")
    print(f"  Threshold    : z ≥ {Z_THRESHOLD}")

    model, tokenizer = load_model()

    print("\n  [1/2] Generating plain text …")
    plain = generate_plain(model, tokenizer, PROMPT)

    print("  [2/2] Generating SynthID watermarked text …")
    watermarked = generate_watermarked(model, tokenizer, PROMPT)

    # Encode generated texts back to token ids for detection
    encode = lambda t: tokenizer.encode(t, add_special_tokens=False)

    plain_z, plain_g, plain_n, plain_p = detect(encode(plain))
    wm_z,    wm_g,    wm_n,    wm_p   = detect(encode(watermarked))

    # ── Per-result blocks ──────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print_result("① Plain (no watermark)", plain,       plain_z, plain_g, plain_n, plain_p)
    print_result("② SynthID watermarked",  watermarked, wm_z,    wm_g,    wm_n,    wm_p)

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Text':<26}  {'Mean g':>8}  {'Z-score':>9}  {'P-value':>10}  Verdict")
    print(f"  {'─'*26}  {'─'*8}  {'─'*9}  {'─'*10}  {'─'*14}")
    for lbl, z, g, n, p in [
        ("Plain (no watermark)", plain_z, plain_g, plain_n, plain_p),
        ("SynthID watermarked",  wm_z,    wm_g,    wm_n,    wm_p),
    ]:
        verdict = "WATERMARKED ✓" if z >= Z_THRESHOLD else "NOT detected ✗"
        print(f"  {lbl:<26}  {g:>8.4f}  {z:>9.3f}  {p:>10.2e}  {verdict}")
    print(f"\n  Null hypothesis: mean_g = 0.5000  (Uniform distribution)")
    print()


if __name__ == "__main__":
    main()
