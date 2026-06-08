"""Local entrypoint: `uv run python scripts/extract_local.py`.

Runs the full extraction pipeline and writes artifacts to ./outputs. Requires
PyTorch (install the GPU extra on Linux, or run on Colab / Modal instead).
"""

from syco_steering.pipeline import run_pipeline

if __name__ == "__main__":
    print(run_pipeline(out_dir="outputs"))
