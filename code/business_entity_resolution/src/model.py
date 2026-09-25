"""JAX pairwise matcher: pure functional logistic regression with JIT training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

Array = jax.Array


@dataclass(frozen=True)
class ModelParams:
    w: Array
    b: Array


def _params_flatten(params: ModelParams):
    return (params.w, params.b), None


def _params_unflatten(_aux, children):
    w, b = children
    return ModelParams(w=w, b=b)


jax.tree_util.register_pytree_node(ModelParams, _params_flatten, _params_unflatten)


def init_params(key: Array, n_features: int) -> ModelParams:
    k_w, _ = random.split(key)
    w = random.normal(k_w, (n_features,), dtype=jnp.float32) * 0.01
    b = jnp.zeros((), dtype=jnp.float32)
    return ModelParams(w=w, b=b)


def predict_logits(params: ModelParams, x: Array) -> Array:
    return x @ params.w + params.b


def predict_proba(params: ModelParams, x: Array) -> Array:
    return jax.nn.sigmoid(predict_logits(params, x))


def binary_nll(params: ModelParams, x: Array, y: Array) -> Array:
    logits = predict_logits(params, x)
    # Stable binary cross-entropy
    return jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _sgd_step(params: ModelParams, x: Array, y: Array, lr: float) -> ModelParams:
    grads = jax.grad(binary_nll)(params, x, y)
    return ModelParams(w=params.w - lr * grads.w, b=params.b - lr * grads.b)


sgd_step = jax.jit(_sgd_step, static_argnames=("lr",))


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    key: Array,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 0.05,
) -> ModelParams:
    """Train with functional SGD; never reuses RNG keys."""
    x_j = jnp.asarray(x, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.float32)
    n = int(x_j.shape[0])
    params = init_params(key, int(x_j.shape[1]))
    if n == 0:
        return params

    key, shuffle_key = random.split(key)
    for epoch in range(epochs):
        shuffle_key, sk = random.split(shuffle_key)
        perm = random.permutation(sk, n)
        xp = x_j[perm]
        yp = y_j[perm]
        for start in range(0, n, batch_size):
            xb = xp[start : start + batch_size]
            yb = yp[start : start + batch_size]
            params = sgd_step(params, xb, yb, lr)
    return params


def score_pairs(params: ModelParams, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    probs = predict_proba(params, jnp.asarray(x, dtype=jnp.float32))
    return np.asarray(probs, dtype=np.float32)


def params_to_numpy(params: ModelParams) -> dict[str, np.ndarray]:
    return {"w": np.asarray(params.w), "b": np.asarray(params.b)}


def params_from_numpy(blob: dict[str, np.ndarray]) -> ModelParams:
    return ModelParams(w=jnp.asarray(blob["w"]), b=jnp.asarray(blob["b"]))


def tune_threshold(
    scores: list[float],
    labels: list[int],
    group_ids: list[str],
    ground_truth: dict[str, set[str]],
    cand_ids: list[str],
    scorer: Callable[[dict[str, set[str]], dict[str, set[str]]], float],
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick threshold maximizing macro F0.5 on a validation split."""
    if grid is None:
        grid = np.linspace(0.35, 0.95, 25)
    best_t, best_s = 0.7, -1.0
    scores_a = np.asarray(scores, dtype=np.float32)
    for t in grid:
        preds: dict[str, set[str]] = {s1: set() for s1 in ground_truth}
        for s, gid, cid in zip(scores_a, group_ids, cand_ids):
            if s >= t:
                preds.setdefault(gid, set()).add(cid)
        # Ensure all GT S1 keys present (incl. singletons)
        for s1 in ground_truth:
            preds.setdefault(s1, set())
        s = scorer(preds, ground_truth)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t, best_s
