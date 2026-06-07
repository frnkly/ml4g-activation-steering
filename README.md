# ML4Good final project

_Work in progress_

## Setup

Install [uv](https://github.com/astral-sh/uv#installation) then run:

```bash
uv sync
```

This will install the "light" stack locally, keeping PyTorch as optional in case
there isn't a supported GPU installed on your machine.

Things that can run locally: tokenization, datasets, sklearn, plotting.

Things that require a GPU: model loading, forward passes, activation steering.

### Running on Google Colab

Colab ships with PyTorch pre-installed, so only the remaining deps are needed:

```python
!pip install transformers accelerate scikit-learn matplotlib
```

Alternatively, if you're on a Linux GPU box with `uv`:

```shell
uv sync --extra gpu
```

### Note on `transformers` without PyTorch

Locally, `import transformers` prints:

```
[transformers] PyTorch was not found. Models won't be available ...
```

This is expected — tokenizers and configs still work fine for preprocessing and
dataset work.
