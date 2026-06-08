# ML4Good final project — Sycophancy steering vector

Phase 1 of replicating the steering-vector extraction half of an activation-steering
paper (arXiv 2604.08169), targeting **sycophancy**. We derive a steering direction
from a binary logistic-regression probe trained on contrastive (sycophantic vs.
honest) activations of `Qwen/Qwen2.5-1.5B-Instruct`, then validate the probe.

See [`IMPLEMENTATION_SPEC.md`](IMPLEMENTATION_SPEC.md) for the full method and
acceptance criteria.

## What it does

1. Loads `meg-tong/sycophancy-eval` (`answer.jsonl`) and filters to the
   **believe-incorrect** template, where affirming the user's wrong answer is
   genuinely *sycophantic* rather than merely *wrong*.
2. For each example, runs a forward pass over `prompt + completion` and
   **mean-pools the hidden states over the completion tokens only**, per layer.
3. Fits a logistic-regression probe per layer (honest = 1, syco = 0), sweeps
   layers, and selects the best.
4. Extracts the steering direction `v̂`, decision boundary `m = −b/‖w‖`, and
   projection statistics.
5. Validates: held-out accuracy + AUROC, a projection histogram, and a
   cosine-with-CAA robustness check; saves plots and `metrics.json`.

## Setup

Install [uv](https://github.com/astral-sh/uv#installation) then run:

```bash
uv sync
```

This installs the **light stack** (transformers, sklearn, numpy, matplotlib,
tqdm) plus dev tools, keeping PyTorch optional in case there isn't a supported GPU
on your machine.

Things that can run locally: tokenization, datasets, sklearn, plotting, the
probe/validate code, and the test suite.

Things that require PyTorch + a GPU: model loading, forward passes, activation
extraction (and Phase 2 steering).

### Note on `transformers` without PyTorch

Locally, `import transformers` prints `PyTorch was not found. Models won't be
available ...`. This is expected — tokenizers and configs still work for
preprocessing and dataset work, and the package keeps torch imports lazy so the
rest of the code stays importable.

## Run the tests

```bash
uv run pytest
```

Includes a CPU smoke test that exercises `layer_sweep`, `extract_direction`, and
`validate` on synthetic Gaussian activations — no model download required.

## Run the full pipeline

### Modal (recommended GPU path)

The workload is a short, intermittent GPU job, so we use [Modal](https://modal.com/docs)
(Python-native, scale-to-zero — you pay only for the seconds the GPU runs). The
model is Apache-2.0/ungated, so **no Hugging Face token is required**. Model
weights and outputs are cached on Modal Volumes.

The Modal CLI is **not** a project dependency (its native deps have no macOS
x86_64 wheels for recent CPython). Install it into the project venv so it can ship
the local `syco_steering` package, and invoke it via `uv run` (it lives in
`.venv/bin`, so a bare `modal` is "command not found" unless the venv is active):

```bash
uv pip install modal               # adds the launcher to this project's venv
uv run modal setup                 # authenticate (first time only)
uv run modal run scripts/app.py    # runs on a T4 GPU
uv run modal volume get syco-outputs / ./outputs   # retrieve artifacts
```

> **`uv sync` removes modal.** Because modal is intentionally not in
> `pyproject.toml`, a later `uv sync` reconciles the venv to the lockfile and
> uninstalls it — just re-run `uv pip install modal` if that happens.

> **Intel macOS note:** `modal` pulls in `cbor2`, whose accelerator is built with
> Rust and has no macOS x86_64 wheel, so pip compiles it from source and needs a
> Rust toolchain (`brew install rust`, or [rustup](https://rustup.rs)) on `PATH`.
> If you'd rather not install Rust, launch the job from Linux or Google Colab
> instead (Apple Silicon and Linux get prebuilt wheels and need none of this).

### Google Colab (no Rust, no GPU box needed)

The easiest GPU path. [`notebooks/extract_sycophancy_colab.ipynb`](notebooks/extract_sycophancy_colab.ipynb)
is **fully self-contained** — every step is plain inline code (no cloning, no
imports from this repo, nothing written to disk during the run). Upload it to
[Colab](https://colab.research.google.com/) (*File → Upload notebook*), set the
runtime to GPU, and *Run all*: it walks through loading data, the model,
activations, the probe, and validation, rendering the plots inline and checking
the acceptance criteria. A final optional cell exports the artifacts if you want
them.

> The notebook restates the pipeline logic standalone, so keep it in sync with
> `src/syco_steering/` if you change the modules.

### Local Linux GPU box

```bash
uv sync --extra gpu
uv run python scripts/extract_local.py   # writes ./outputs
```

## Artifacts (`outputs/`)

- `steering_vector.npz` — `v_hat`, `steering_vector`, `m`, `mu_pos`, `sig_pos`,
  `delta_mu`, `best_layer`, `model`.
- `metrics.json` — layer-sweep accuracies, `best_layer`, `test_acc`, `auroc`,
  `caa_cosine`, `n_pairs` (kept in version control).
- `layer_acc.png`, `projection_hist.png`.

Hidden-state index convention: `output_hidden_states=True` returns `num_layers + 1`
tensors, where index `0` is the embedding output and index `l` is the output of
decoder layer `l-1`. `best_layer` is recorded as a hidden_states index; Phase 2
will hook `model.model.layers[best_layer - 1]`.

## Phase 2 (not built yet)

> Phase 2 will load `steering_vector.npz` and register a forward hook on
> `model.model.layers[best_layer - 1]` implementing SwFC / StTP / StMP, gated on
> the per-token projection `v̂·hₜ` relative to the boundary `m`. The steering
> hooks, inference-time intervention, and the SycophancyEval evaluation harness
> are out of scope for Phase 1.
