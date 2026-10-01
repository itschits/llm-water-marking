"""
OPT-125M  ×  STA-1  (Sampling One Then Accepting)
==================================================
Completely self-contained — no dependency on src/.

Model
-----
facebook/opt-125m  (Meta OPT family, 125 M parameters)
  - Different architecture from GPT-2 (uses causal attention with
    learned positional embeddings and pre-norm)
  - Same HuggingFace AutoModelForCausalLM interface
  - No authentication required; CPU-friendly

Algorithm  (identical to experiments/gpt2/sta1.py)
---------------------------------------------------
STA-1 embeds a watermark via REJECTION SAMPLING without ever touching
the model's logit distribution:

  At each step t:
  1. Hash (SEED, x_{t-1}) with SHA-256 → shuffle vocabulary → take the
     first ⌊γ·|V|⌋ entries as the GREEN LIST for this context.
  2. Sample a candidate c from the UNMODIFIED distribution p(·|x_{<t}).
  3. If c ∈ green list → accept and emit.
     Else              → reject, sample once more, and accept regardless.

Detection counts green tokens and applies a one-sided z-test:
  z = (green_count − γ·T) / √(T·γ·(1−γ))

Run
---
    python experiments/opt_125m/sta1.py
"""

import hashlib
import math
import textwrap

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────
MODEL_ID       = "facebook/opt-125m"
PROMPT         = "Artificial intelligence is transforming the world because"
MAX_NEW_TOKENS = 120
TEMPERATURE    = 1.0

# STA-1 watermark parameters
SEED           = 42      # secret shared between generator and detector
GAMMA          = 0.5     # green-list fraction  (half the vocabulary)
Z_THRESHOLD    = 4.0     # detection threshold  (FPR ≈ 3e-5)


# ── Green-list construction ────────────────────────────────────────────

def get_green_list(prev_token_id: int, vocab_size: int) -> frozenset:
    """
    Deterministically derive the green list for a given previous token.
    Fully determined by (SEED, prev_token_id); identical on every call.
    """
    raw      = f"{SEED}:{prev_token_id}".encode()
    rng_seed = int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")
    rng      = np.random.default_rng(rng_seed)
    perm     = rng.permutation(vocab_size)
    size     = max(1, int(GAMMA * vocab_size))
    return frozenset(int(x) for x in perm[:size])


# ── Sampling helpers ───────────────────────────────────────────────────

def _sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """One multinomial draw from softmax(logits / temperature)."""
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)


def rejection_sample(logits: torch.Tensor, green_list: frozenset,
                     temperature: float = 1.0) -> torch.Tensor:
    """
    STA-1 core sampler.

    Draw from the UNMODIFIED distribution.
    If the token is on the green list → accept.
    Otherwise → reject and sample exactly once more, accepting regardless.
    Logits are NEVER modified.
    """
    token = _sample(logits, temperature)
    if token.item() in green_list:
        return token
    return _sample(logits, temperature)  # one resample, accept regardless


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
    return model, tokenizer, model.config.vocab_size


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


def generate_watermarked(model, tokenizer, vocab_size: int, prompt: str) -> str:
    """STA-1 watermarked generation — manual autoregressive loop."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]
    generated  = input_ids.clone()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(generated).logits[0, -1, :].float()   # unmodified
            prev   = generated[0, -1].item()
            gl     = get_green_list(prev, vocab_size)
            tok    = rejection_sample(logits, gl, TEMPERATURE)
            generated = torch.cat([generated, tok.view(1, 1)], dim=1)
            if tok.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)


# ── Detection ──────────────────────────────────────────────────────────

def detect(token_ids: list, vocab_size: int):
    """
    Count green tokens, compute z-score and p-value.

    H₀: green count ~ Binomial(T, γ)  (no watermark)
    z  = (green_count − γT) / √(T·γ·(1−γ))
    """
    total = len(token_ids) - 1
    if total <= 0:
        return 0.0, 0, 0, 1.0

    green_count = sum(
        1
        for i in range(1, len(token_ids))
        if token_ids[i] in get_green_list(token_ids[i - 1], vocab_size)
    )

    expected = GAMMA * total
    std      = math.sqrt(total * GAMMA * (1 - GAMMA))
    z        = (green_count - expected) / std if std > 0 else 0.0
    p_value  = float(1.0 - stats.norm.cdf(z))
    return z, green_count, total, p_value


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
                 green: int, total: int, p_val: float) -> None:
    detected = z >= Z_THRESHOLD
    verdict  = "WATERMARKED ✓" if detected else "NOT detected ✗"
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"{'─'*72}")
    print(_wrap(text))
    print()
    print(f"  Green tokens : {green} / {total}  ({green / total * 100:.1f} %)")
    print(f"  Z-score      : {z:.4f}   {_bar(z)}")
    print(f"  P-value      : {p_val:.2e}")
    print(f"  Decision     : z ≥ {Z_THRESHOLD} →  {verdict}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  OPT-125M  ×  STA-1  (Sampling One Then Accepting)")
    print("=" * 72)
    print(f"\n  Model    : {MODEL_ID}")
    print(f"  Prompt   : {PROMPT!r}")
    print(f"  Max tok  : {MAX_NEW_TOKENS}   temperature: {TEMPERATURE}")
    print(f"  γ        : {GAMMA}   seed: {SEED}")
    print(f"  Threshold: z ≥ {Z_THRESHOLD}")

    model, tokenizer, vocab_size = load_model()

    print("\n  [1/2] Generating plain text …")
    plain = generate_plain(model, tokenizer, PROMPT)

    print("  [2/2] Generating STA-1 watermarked text …")
    watermarked = generate_watermarked(model, tokenizer, vocab_size, PROMPT)

    # Encode generated texts back to token ids for detection
    encode = lambda t: tokenizer.encode(t, add_special_tokens=False)

    plain_z, plain_g, plain_tot, plain_p = detect(encode(plain),       vocab_size)
    wm_z,    wm_g,    wm_tot,    wm_p   = detect(encode(watermarked), vocab_size)

    # ── Per-result blocks ──────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print_result("① Plain (no watermark)", plain,       plain_z, plain_g, plain_tot, plain_p)
    print_result("② STA-1 watermarked",    watermarked, wm_z,    wm_g,    wm_tot,    wm_p)

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Text':<26}  {'Green':>12}  {'Z-score':>9}  {'P-value':>10}  Verdict")
    print(f"  {'─'*26}  {'─'*12}  {'─'*9}  {'─'*10}  {'─'*14}")
    for lbl, z, g, tot, p in [
        ("Plain (no watermark)", plain_z, plain_g, plain_tot, plain_p),
        ("STA-1 watermarked",    wm_z,    wm_g,    wm_tot,    wm_p),
    ]:
        verdict = "WATERMARKED ✓" if z >= Z_THRESHOLD else "NOT detected ✗"
        pct     = f"{g}/{tot} ({g / tot * 100:.0f} %)"
        print(f"  {lbl:<26}  {pct:>12}  {z:>9.3f}  {p:>10.2e}  {verdict}")
    print()


if __name__ == "__main__":
    main()
