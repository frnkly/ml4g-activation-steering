"""Modal app for GPU runs: `modal run scripts/app.py`.

First-time setup:
    uv pip install modal       # launcher CLI (not a project dependency)
    uv run modal setup         # authenticate

Run:        uv run modal run scripts/app.py
Retrieve:   uv run modal volume get syco-outputs / ./outputs

(modal lives in .venv/bin, so invoke it via `uv run` unless the venv is active.)
"""

import modal

app = modal.App("syco-steering")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "tqdm",
    )
    # `add_local_python_source` takes a module name (not a path) and resolves it
    # via the local import machinery, so the src-layout `syco_steering` package
    # must be importable locally (it is, via `uv sync`).
    .add_local_python_source("syco_steering")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("syco-outputs", create_if_missing=True)


@app.function(
    # T4 (16 GB) is Modal's smallest/cheapest GPU and is plenty for a 1.5B model
    # in fp16. Bump to "L4" if you ever hit out-of-memory.
    gpu="T4",
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs},
    # The pipeline now generates ~120 on-policy responses before extraction,
    # so give it more headroom than a pure forward-pass job needs.
    timeout=60 * 45,
)
def run():
    from syco_steering.pipeline import run_pipeline

    metrics = run_pipeline(out_dir="/outputs")
    outputs.commit()
    return metrics


@app.local_entrypoint()
def main():
    print(run.remote())
