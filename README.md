# ML4Good final project — Sycophancy steering vector

Replication of the method of **"Activation Steering for Aligned Open-ended
Generation without Sacrificing Coherence"** (Herbster et al., arXiv 2604.08169),
adapted to **sycophancy** as the target trait and scaled down to
`Qwen/Qwen2.5-1.5B-Instruct`.

## Method

Following the paper's design:

1. **Induced misalignment.** A *sycophancy-inducing* system prompt plays the role
   of the paper's malicious system prompt; an *honesty-inducing* system prompt is
   the aligned side. The contrastive texts are the model's **own generated
   responses** under each, to questions from `meg-tong/sycophancy-eval`
   (`answer.jsonl`, filtered to the **believe-incorrect** template, where the user
   states a wrong belief — so affirming it is genuinely *sycophantic* rather than
   merely wrong). Training on on-policy generations means the probe's activations
   match the distribution the steering gate sees at inference.
2. **Response-averaged probe (paper Eq. 1).** Each response contributes one
   training example: the *mean* hidden state over its response tokens at layer
   ℓ. A logistic-regression probe on these pooled embeddings gives the steering
   direction `v̂` (normalized weights), the decision boundary `m = −b/‖w‖`
   (the paper's Alg. A.1 rescaling reduces to exactly this), and the pooled
   projection stats `μ⁺, σ⁺, Δμ`. The inference-time StTP/StMP gate still
   compares *individual token* projections to `m` — this calibration transfers
   because averaging shrinks the within-class variance but leaves the class
   means unchanged, so the pooled boundary sits between the same means the
   (wider) per-token distributions straddle. A token-level probe sweep is kept
   as a cheap layer-shortlist heuristic and for gate diagnostics.
3. **Steering** (notebook Part B): a forward hook on the extraction layer
   implements the paper's three methods in the paper's parameterization —
   **SwFC** (`h' = h + α·Δμ·v̂`; α = 1 is one unit of class separation),
   **StTP** (tokens with projection `ρ < m` are set to the target `μ⁺ + α·σ⁺`;
   σ⁺ is the pooled std, so the paper's α grid runs to 36), and **StMP**
   (tokens with `ρ < m` are reflected across the boundary:
   `h' = h + 2α(m − ρ)·v̂`). Steering position defaults to the paper's
   **all-token** mode (every position, prompt prefill included — the mode
   behind all of the paper's headline results); response-only steering is kept
   as the ablation, which the paper shows recovers roughly half the trait
   score (Table C.1).
4. **Validation / evaluation:** held-out pooled-probe accuracy + AUROC,
   pooled + per-token projection histograms, a CAA-cosine consistency check,
   a gate-firing diagnostic on fresh generations, an α sweep over the paper's
   coefficient grids on prompts disjoint from the final eval, and a held-out
   comparison against both baselines — scored with string proxies plus two of
   the paper's judge-free metrics: cross-entropy under the unsteered model
   conditioned on the aligned prompt (§C.3) and embedding similarity to the
   aligned baseline's responses (§C.5).

See [`docs/paper-replication-review.md`](docs/paper-replication-review.md) for
a detailed comparison of this implementation against the paper, including the
reasoning behind each of the choices above.

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

Things that require PyTorch + a GPU: model loading, generation, activation
extraction, and steering.

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
`validate` on synthetic per-token Gaussian activations (with group structure) —
no model download required.

## Run the full pipeline

### Google Colab (recommended; runs both halves)

[`notebooks/sycophancy_steering.ipynb`](notebooks/sycophancy_steering.ipynb)
is **fully self-contained** — every step is plain inline code (no cloning, no
imports from this repo, nothing written to disk during the run). Upload it to
[Colab](https://colab.research.google.com/) (*File → Upload notebook*), set the
runtime to GPU, and *Run all* (roughly 25–35 minutes on a free T4). **Part A**
generates the contrastive responses, extracts per-token activations, fits the
probe, and validates the direction; **Part B** hooks the extraction layer,
verifies the StTP/StMP gate actually fires on misaligned generations, sweeps α,
and measures the steering effect on held-out prompts against both baselines.
The model is ungated, so no Hugging Face token is needed.

> The notebook restates the pipeline logic standalone, so keep it in sync with
> `src/syco_steering/` if you change the modules. (The `src/` pipeline covers
> the extraction-and-validation half; the steering hook and eval live in the
> notebook's Part B.)

### Modal (GPU extraction pipeline)

The extraction half can also run as a short scriptable GPU job on
[Modal](https://modal.com/docs) (Python-native, scale-to-zero). Model weights and
outputs are cached on Modal Volumes.

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

### Local Linux GPU box

```bash
uv sync --extra gpu
uv run python scripts/extract_local.py   # writes ./outputs
```

## Artifacts (`outputs/`)

- `steering_vector.npz` — `v_hat`, `steering_vector`, `m`, `mu_pos`, `sig_pos`,
  `delta_mu`, `best_layer`, `model`. The projection statistics are in
  response-averaged units (paper Eq. 1) — coefficients like StTP's α multiply
  the *pooled* σ⁺, which is why the paper's α grid runs to 36.
- `responses.json` — the generated contrastive training responses, for inspection.
- `metrics.json` — layer-sweep accuracies, `best_layer`, `test_acc`, `auroc`,
  `caa_cosine`, token counts (kept in version control).
- `layer_acc.png`, `projection_hist.png`.

Hidden-state index convention: `output_hidden_states=True` returns `num_layers + 1`
tensors, where index `0` is the embedding output and index `l` is the output of
decoder layer `l-1`. `best_layer` is recorded as a hidden_states index; the
steering hook attaches to `model.model.layers[best_layer - 1]`.

## Known limitations vs. the paper

- The paper runs Llama-3.3-70B-Instruct and Qwen3.6-27B; linear trait structure
  is cleaner at scale, so expect noisier results from a 1.5B model.
- Trait and coherence are scored with string/repetition proxies plus two of the
  paper's judge-free metrics (cross-entropy vs. the aligned model, embedding
  similarity to the aligned baseline) instead of the paper's LLM judge
  (GPT-oss-120B), ELO tournament, and capability suite (MMLU / MT-Bench /
  AlpacaEval).
- Decoding is greedy at 60–80 tokens; the paper samples at T=0.6 / top-p=0.9
  up to 1024 tokens, where the long-generation repetition pathology that
  separates StTP/StMP from SwFC actually shows up.
- The steering layer is the probe-shortlisted layer; the paper picks an
  operating point over a full layer × coefficient × position grid judged on
  trait + coherence (its own selection rule, and the highest-value next step).
- Both sides of the contrast use a single fixed system-prompt pair, so the
  probe may partly encode "which system prompt is in context" rather than the
  trait itself. The paper mitigates (not eliminates) this by varying the
  contrastive prompts across scenarios — 5 paraphrase variants for
  dismissiveness, per-scenario prompts for dishonesty.
