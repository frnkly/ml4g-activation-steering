"""Model loading, on-policy response generation, and per-token activation extraction.

torch / transformers are imported lazily inside the functions so the rest of the
package (data, probe, validate) stays importable on machines without PyTorch.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def load_model(name: str):
    """Return ``(model, tok)``. fp16 on cuda if available, else fp32 on cpu.

    Calls ``model.eval()`` and sets ``tok.pad_token = tok.eos_token`` if missing.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def generate_responses(
    model,
    tok,
    prompts: list[dict],
    system: str,
    max_new: int,
    batch_size: int = 8,
) -> list[str]:
    """Greedy-decode one response per prompt under the given system prompt.

    These on-policy generations are the probe's training texts (the paper trains
    on the model's own responses under aligned vs. misalignment-inducing system
    prompts, so the activations match the inference-time distribution).
    """
    import torch

    texts = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": p["user"]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]
    responses: list[str] = []
    old_side = tok.padding_side
    tok.padding_side = "left"  # left padding so every row ends at the same position
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="generate"):
            enc = tok(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            responses.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tok.padding_side = old_side
    return responses


def response_token_acts(model, tok, system: str, user: str, response: str) -> np.ndarray:
    """Forward pass over system+user+response; return the hidden states of every
    RESPONSE token, per layer: array of shape ``(T, num_layers + 1, hidden)`` in
    float16.

    The probe is trained and the steering boundary is applied PER TOKEN, so no
    pooling happens here — pooling would calibrate the boundary ``m`` and the
    stats ``mu_pos`` / ``sig_pos`` on a tighter distribution than the raw token
    states the inference-time gate sees, and the gate would (almost) never fire.

    The prompt boundary is the length of the chat-templated prompt rendered with
    ``add_generation_prompt=True``; everything after it is the response.
    """
    import torch

    def _ids(out):
        # Recent transformers return a BatchEncoding dict from apply_chat_template;
        # older versions return a bare tensor. Normalize to the input_ids tensor.
        ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
        return ids.to(model.device)

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    with torch.no_grad():
        p_ids = _ids(
            tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        )
        f_ids = _ids(
            tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": response}],
                add_generation_prompt=False,
                return_tensors="pt",
            )
        )
        plen = p_ids.shape[1]
        assert torch.equal(f_ids[0, :plen], p_ids[0]), (
            "chat template is not prefix-stable; cannot locate the response tokens"
        )
        out = model(f_ids, output_hidden_states=True)
        # (L+1, T, d) -> (T, L+1, d); response tokens only.
        hs = torch.stack(out.hidden_states, 0)[:, 0, plen:, :]
        return hs.permute(1, 0, 2).to(torch.float16).cpu().numpy()


def extract_token_acts(
    model, tok, prompts: list[dict], responses: list[str], system: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token activations for each (prompt, response) pair under ``system``.

    Returns ``(X, groups)`` where ``X`` has shape ``(T_total, num_layers + 1,
    hidden)`` (float16) and ``groups[t]`` is the index of the prompt that token
    ``t`` came from — used for group-aware train/test splits so tokens from the
    same response never straddle the split.
    """
    chunks, groups = [], []
    for i, (p, resp) in enumerate(
        tqdm(list(zip(prompts, responses)), desc="activations")
    ):
        acts = response_token_acts(model, tok, system, p["user"], resp)
        if len(acts) == 0:  # empty generation; skip
            continue
        chunks.append(acts)
        groups.append(np.full(len(acts), i, dtype=np.int64))
    return np.concatenate(chunks, axis=0), np.concatenate(groups, axis=0)
