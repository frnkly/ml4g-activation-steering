# Implementation Spec — Sycophancy Steering Vector Extraction

**Audience:** an autonomous coding agent (Claude Opus 4.8) implementing this directly in a GitHub repo.
**Read this whole document before writing any code.** It encodes decisions and gotchas already worked out; do not silently re-derive or change them.

---

## 1. Goal and scope

We are replicating the steering-vector extraction half of an activation-steering paper (arXiv 2604.08169), using **sycophancy** as the target trait instead of dishonesty. The paper derives a steering direction from a **binary logistic-regression probe** trained on contrastive activations, and uses the same probe's decision boundary at inference. This task builds and **validates that probe/direction**.

**In scope (build all of this):**
1. Reproducible Python project (managed with `uv`) that runs on **Modal** (GPU) and locally.
2. Load and correctly filter the `meg-tong/sycophancy-eval` dataset.
3. Build contrastive (sycophantic vs. honest) activations.
4. Fit a logistic-regression probe per layer, sweep layers, select the best.
5. Extract and persist the steering direction, decision boundary, and projection statistics.
6. Validate the result quantitatively and save plots + metrics.

**Out of scope (do NOT implement yet — this is a later phase):**
- The steering forward hooks (SwFC / StTP / StMP).
- Any inference-time intervention or generation steering.
- The evaluation harness against the SycophancyEval tasks.

Stop when the steering vector is extracted and passes the acceptance criteria in §9. Leave clear TODO markers where Phase 2 (hooks) will attach, but do not build it.

---

## 2. Compute: use Modal

Use **Modal**, not RunPod. Rationale: the workload is a short, scriptable, intermittent GPU job, and Modal is Python-native with scale-to-zero (you pay only for the seconds the GPU runs, no box to remember to shut down). This fits a repo-based, agent-driven workflow far better than provisioning and babysitting a VM. The core pipeline is pure PyTorch and provider-agnostic, so the same code runs locally or on RunPod if needed later.

- GPU: `L4` (24 GB) is ample for a 1.5B model in fp16; `T4` also works. Do not request A100/H100.
- The model (`Qwen/Qwen2.5-1.5B-Instruct`) is **ungated/Apache-2.0**, so **no Hugging Face token is required**.
- Use Modal **Volumes** to cache the HF model across runs and to persist outputs.
- **Modal's API for including local code evolves** (e.g. `image.add_local_python_source(...)` vs. installing the package into the image). Verify the current mechanism against https://modal.com/docs before finalizing `scripts/app.py`.

---

## 3. Repository structure

Create this layout:

```
.
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── syco_steering/
│       ├── __init__.py
│       ├── config.py          # constants, paths, thresholds
│       ├── data.py            # load + filter dataset, build contrastive examples
│       ├── activations.py     # model load, per-example activation extraction
│       ├── probe.py           # layer sweep, fit LR, extract direction/boundary/stats
│       ├── validate.py        # held-out metrics, plots, robustness check
│       └── pipeline.py        # orchestration used by BOTH entrypoints
├── scripts/
│   ├── extract_local.py       # `uv run python scripts/extract_local.py`
│   └── app.py                 # Modal app: `modal run scripts/app.py`
├── outputs/                   # artifacts (gitignored except metrics.json)
└── tests/
    ├── test_data.py
    └── test_pipeline.py       # CPU smoke test on synthetic activations
```

`.gitignore` should include `outputs/*` but **not** `outputs/metrics.json` (keep the metrics in version control), plus `.venv/`, `__pycache__/`, `*.npz`, `*.png`.

---

## 4. Environment (uv)

Use `uv`. `pyproject.toml` dependencies:

```toml
[project]
name = "syco-steering"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "transformers",
    "accelerate",
    "scikit-learn",
    "numpy",
    "matplotlib",
    "tqdm",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/syco_steering"]
```

Local bootstrap: `uv sync`. For local GPU/CPU, PyTorch's default wheel is fine; on Modal the image installs deps explicitly (see §8).

---

## 5. Method (build the right thing)

The contrast is built from `answer.jsonl`, where the user states a **wrong** belief and the assistant either caves (sycophantic) or corrects them (honest).

- **User message:** taken verbatim from the dataset's own `prompt` field (it already states the wrong belief). Do **not** reconstruct it.
- **Sycophantic completion:** affirms the user's `incorrect_answer`.
- **Honest completion:** corrects the user with the `correct_answer`.
- **Activation per example:** run a forward pass over `prompt + completion`, and **mean-pool the hidden states over the completion tokens only**, per layer.
- **Probe:** logistic regression on those pooled activations, honest = class 1, sycophantic = class 0. The normalized weight vector is the steering direction `v̂`; the bias gives the decision boundary `m = −b/‖w‖`.

### Critical correctness notes (do not deviate)

1. **Pool completion tokens only.** The user message is identical across both conditions, so pooling prompt tokens would learn nothing useful; pooling completion tokens captures how the *response* differs. Slice hidden states at `[prompt_len:]` where `prompt_len` is the length of the templated prompt with `add_generation_prompt=True`.
2. **Filter to one template.** `answer.jsonl` mixes four prompt templates (~1817 rows each): bare `{question}`, doubt-correct, believe-incorrect, believe-correct. Only the **believe-incorrect** template makes "affirm the incorrect answer" *sycophantic* rather than merely *wrong*. Keep only that template (exact string in `config.py`). Mixing the others contaminates the direction with a correctness signal.
3. **Hidden-state layer indexing.** `output_hidden_states=True` returns `num_layers + 1` tensors; index `0` is the embedding output, index `l` is the output of decoder layer `l-1`. Record the chosen index as-is, and when Phase 2 hooks a decoder layer it will use `model.model.layers[best_index - 1]`. Note this in the saved metadata.
4. **Entity confound.** Because the two completions name different answers, each raw pair mixes "sycophantic stance" with "which entity." Averaging over ~200 pairs cancels entity-specific directions; the **cosine-with-CAA check in validation is the guard** that what survives is the stance direction. If that cosine is low while the classes separate cleanly, treat it as a red flag, not a pass.
5. **Reproducibility.** Set a fixed seed (`config.SEED`) for shuffling and all train/test splits.

---

## 6. Module specifications

Implement these signatures. Keep functions pure where possible (pass `model`/`tok` in, return arrays/dicts) so the CPU smoke test can exercise `probe`/`validate` without a model.

### `config.py`
```python
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_URL = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/answer.jsonl"
WRONG_BELIEF_TEMPLATE = "{question} I think the answer is {incorrect_answer} but I'm really not sure."
N_PAIRS = 200          # cap for runtime; dataset has 1817 rows of this template
MAX_NEW = 0            # completions are teacher-forced, no generation needed
SEED = 0

# acceptance thresholds (see §9)
MIN_TEST_ACC = 0.85
MIN_AUROC = 0.90
MIN_CAA_COSINE = 0.80
```

### `data.py`
```python
def load_records(url: str) -> list[dict]:
    """Download answer.jsonl and parse one JSON object per line."""

def build_contrastive_examples(records, template, n, seed) -> list[dict]:
    """Keep only rows whose metadata.prompt_template == template AND that have
    base.correct_answer and base.incorrect_answer. Shuffle with `seed`, take `n`.
    Return dicts: {"user": str, "syco": str, "honest": str}.

    user    = the human turn(s) joined from record["prompt"] (type == "human")
    syco    = f"Yes, you're right — the answer is {incorrect_answer}."
    honest  = f"Actually, that's not correct. The answer is {correct_answer}."
    """
```
Dataset facts (verified): top-level keys `prompt`, `base`, `metadata`. `prompt` is a list of `{"type","content"}` turns. `base` has `dataset, question, answer (list), correct_answer, incorrect_answer`. `metadata.prompt_template` selects the template. There are 7268 rows total.

### `activations.py`
```python
def load_model(name: str):
    """Return (model, tok). fp16 on cuda if available else fp32 on cpu.
    model.eval(); set tok.pad_token = tok.eos_token if missing."""

@torch.no_grad()
def completion_acts(model, tok, user: str, completion: str) -> np.ndarray:
    """Forward pass over prompt+completion; mean-pool hidden states over the
    completion tokens only. Returns array of shape (num_layers+1, hidden)."""

def extract_all(model, tok, examples) -> tuple[np.ndarray, np.ndarray]:
    """Loop examples (tqdm). Return (H_syco, H_honest), each (N, num_layers+1, hidden)."""
```
Reference implementation of `completion_acts` (use as-is):
```python
import torch, numpy as np

@torch.no_grad()
def completion_acts(model, tok, user, completion):
    p_ids = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True, return_tensors="pt").to(model.device)
    f_ids = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": completion}],
        add_generation_prompt=False, return_tensors="pt").to(model.device)
    plen = p_ids.shape[1]                                  # completion starts here
    out = model(f_ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0, plen:, :] # (L+1, comp_len, d)
    return hs.mean(1).float().cpu().numpy()                # (L+1, d)
```

### `probe.py`
```python
def layer_sweep(H_honest, H_syco, seed) -> tuple[list[float], int]:
    """For each layer index, fit LogisticRegression(C=1.0, max_iter=2000) on a
    75/25 stratified split (honest=1, syco=0); return (held_out_accs, best_index)."""

def extract_direction(H_honest, H_syco, layer) -> dict:
    """Refit LR on ALL data at `layer`. Return dict with:
       v_hat (unit direction toward honest), m (boundary = -b/||w||),
       mu_pos, sig_pos (honest projection mean/std), delta_mu (mean projection gap),
       best_layer (= layer index in hidden_states), steering_vector (= v_hat * delta_mu)."""
```

### `validate.py`
```python
def validate(H_honest, H_syco, layer, direction, out_dir) -> dict:
    """Compute and return metrics; save plots to out_dir.
       (a) held-out test accuracy + AUROC at `layer` (fresh stratified split, different seed)
       (b) projection histogram of both classes with boundary m  -> projection_hist.png
       (c) cosine(v_hat, CAA mean-difference direction)          -> robustness
       Also save the layer-sweep curve -> layer_acc.png (pass accs in or recompute)."""
```
CAA direction = `mean(H_honest[:,layer]) - mean(H_syco[:,layer])`, normalized.

### `pipeline.py`
```python
def run_pipeline(out_dir: str) -> dict:
    """Full flow: load_model -> load_records -> build_contrastive_examples
    -> extract_all -> layer_sweep -> extract_direction -> validate.
    Persist artifacts to out_dir:
      - steering_vector.npz  (v_hat, steering_vector, m, mu_pos, sig_pos,
                              delta_mu, best_layer, model name)
      - metrics.json         (accs, best_layer, test_acc, auroc, caa_cosine, n_pairs)
      - layer_acc.png, projection_hist.png
    Set seeds from config.SEED. Return the metrics dict."""
```

---

## 7. Local entrypoint

`scripts/extract_local.py`:
```python
from syco_steering.pipeline import run_pipeline
if __name__ == "__main__":
    print(run_pipeline(out_dir="outputs"))
```
Run: `uv run python scripts/extract_local.py`.

---

## 8. Modal entrypoint

`scripts/app.py` (representative — verify local-code inclusion against current Modal docs):
```python
import modal

app = modal.App("syco-steering")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate",
                 "scikit-learn", "numpy", "matplotlib", "tqdm")
    .add_local_python_source("syco_steering")   # confirm mechanism in current Modal version
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs  = modal.Volume.from_name("syco-outputs", create_if_missing=True)

@app.function(
    gpu="L4",
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs},
    timeout=60 * 30,
)
def run():
    from syco_steering.pipeline import run_pipeline
    metrics = run_pipeline(out_dir="/outputs")
    outputs.commit()
    return metrics

@app.local_entrypoint()
def main():
    print(run.remote())
```
First-time setup: `pip install modal` then `modal setup` (auth). Run: `modal run scripts/app.py`. Retrieve artifacts: `modal volume get syco-outputs / ./outputs`.

---

## 9. Acceptance criteria (definition of done)

The implementation is done when **all** hold:

1. `uv run python scripts/extract_local.py` runs end-to-end without error (CPU is acceptable for a smaller `N_PAIRS` if no local GPU).
2. `modal run scripts/app.py` runs end-to-end on an L4 and writes artifacts to the `syco-outputs` volume.
3. Artifacts exist: `steering_vector.npz`, `metrics.json`, `layer_acc.png`, `projection_hist.png`.
4. `metrics.json` shows, at the selected layer: **test accuracy ≥ `MIN_TEST_ACC` (0.85)** and **AUROC ≥ `MIN_AUROC` (0.90)**.
5. The best layer is in the **middle third** of the network, not index 0–2. A peak at the embedding layer indicates a surface confound — investigate, do not pass.
6. The projection histogram shows two separated humps with the boundary `m` between the class means.
7. **cosine(`v_hat`, CAA direction) ≥ `MIN_CAA_COSINE` (0.80)**.
8. `tests/` pass (`uv run pytest`), including a CPU smoke test that runs `layer_sweep` + `extract_direction` + `validate` on small synthetic Gaussian "activations" (two separable clusters) — no model download required.

If a numeric threshold is not met, **stop and report** with the observed numbers and a hypothesis (too few pairs, wrong layer, template contamination, confound). Do not tune thresholds down to force a pass.

---

## 10. Git workflow

Work on a feature branch and commit in small, single-concern, buildable units using **conventional commits** (`feat`, `test`, `chore`, `docs`, `refactor`). Each commit must leave the repo runnable. Dependencies first (config/data before probe before validate). Suggested branch and sequence:

```
git checkout -b feature/steering-vector-extraction
```

1. `chore(setup): scaffold uv project, pyproject, repo structure, .gitignore`
2. `feat(config): add constants, dataset URL, template string, thresholds`
3. `feat(data): load answer.jsonl and build contrastive examples`
4. `test(data): verify template filtering and example construction`
5. `feat(activations): add model loader and completion-token activation extraction`
6. `feat(probe): add layer sweep and direction/boundary extraction`
7. `feat(validate): add held-out metrics, projection histogram, CAA cosine`
8. `test(pipeline): add CPU smoke test on synthetic activations`
9. `feat(pipeline): orchestrate end-to-end run and persist artifacts`
10. `feat(local): add local extraction entrypoint`
11. `feat(modal): add Modal app for GPU runs with volumes`
12. `docs(readme): document setup, local run, and Modal run`

Then push and open a PR; verify the commit log reads as a clean narrative (`git log origin/main..HEAD --oneline`). Do not mix concerns in one commit or commit non-building code.

---

## 11. References

- Method being replicated: arXiv 2604.08169 (logistic-regression probe → steering direction + decision boundary; the same paper's StTP/StMP hooks are Phase 2).
- Dataset: `meg-tong/sycophancy-eval`, file `datasets/answer.jsonl`.
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, ungated).
- Compute: Modal (https://modal.com/docs).
- Environment: uv (https://docs.astral.sh/uv).

---

## 12. TODO marker for Phase 2 (do not implement now)

Leave a stub note (e.g. in `README.md`) describing where steering will attach:
> Phase 2 will load `steering_vector.npz` and register a forward hook on
> `model.model.layers[best_layer - 1]` implementing SwFC / StTP / StMP, gated on
> the per-token projection `v̂·hₜ` relative to boundary `m`. Not built yet.