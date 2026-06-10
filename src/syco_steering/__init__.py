"""Sycophancy steering-vector extraction (arXiv 2604.08169, adapted to sycophancy).

Derives a steering direction from a binary logistic-regression probe trained on
per-token activations of the model's own responses generated under a
sycophancy-inducing vs. an honesty-inducing system prompt. See README.md for the
method and acceptance criteria.
"""

__all__ = ["config"]
