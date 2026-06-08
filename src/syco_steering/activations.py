"""Model loading and per-example activation extraction.

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

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def completion_acts(model, tok, user: str, completion: str) -> np.ndarray:
    """Forward pass over prompt+completion; mean-pool hidden states over the
    completion tokens only. Returns array of shape ``(num_layers + 1, hidden)``.

    The prompt boundary is the length of the chat-templated prompt rendered with
    ``add_generation_prompt=True``; everything after it is the completion.
    """
    import torch

    def _ids(out):
        # Recent transformers return a BatchEncoding dict from apply_chat_template;
        # older versions return a bare tensor. Normalize to the input_ids tensor.
        ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
        return ids.to(model.device)

    with torch.no_grad():
        p_ids = _ids(
            tok.apply_chat_template(
                [{"role": "user", "content": user}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        )
        f_ids = _ids(
            tok.apply_chat_template(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": completion},
                ],
                add_generation_prompt=False,
                return_tensors="pt",
            )
        )
        plen = p_ids.shape[1]  # completion starts here
        out = model(f_ids, output_hidden_states=True)
        # (L+1, comp_len, d) — slice batch item 0 and completion tokens only.
        hs = torch.stack(out.hidden_states, 0)[:, 0, plen:, :]
        return hs.mean(1).float().cpu().numpy()  # (L+1, d)


def extract_all(model, tok, examples) -> tuple[np.ndarray, np.ndarray]:
    """Loop over examples. Return ``(H_syco, H_honest)``, each of shape
    ``(N, num_layers + 1, hidden)``."""
    syco, honest = [], []
    for ex in tqdm(examples, desc="activations"):
        syco.append(completion_acts(model, tok, ex["user"], ex["syco"]))
        honest.append(completion_acts(model, tok, ex["user"], ex["honest"]))
    return np.stack(syco), np.stack(honest)
