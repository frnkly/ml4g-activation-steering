"""Modal app for GPU runs: `modal run scripts/app.py`.

First-time setup:
    uv pip install modal       # launcher CLI (not a project dependency)
    modal setup                # authenticate

Run:        modal run scripts/app.py
Retrieve:   modal volume get syco-outputs / ./outputs
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
