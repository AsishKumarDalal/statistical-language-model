"""Statistical utility functions for the language model."""

import numpy as np
from typing import Optional


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    x_scaled = x / temperature
    x_shifted = x_scaled - np.max(x_scaled, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
    return x_shifted - log_sum_exp


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    log_probs = log_softmax(logits)
    batch_size = targets.shape[0]
    target_log_probs = log_probs[np.arange(batch_size), targets]

    if mask is not None:
        target_log_probs = target_log_probs * mask
        return -np.sum(target_log_probs) / np.sum(mask)

    return -np.mean(target_log_probs)


def perplexity(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    ce = cross_entropy(logits, targets, mask)
    return np.exp(ce)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(p * np.log(p / q))


def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
    p = np.clip(p, epsilon, 1.0)
    p = p / np.sum(p)
    return -np.sum(p * np.log(p))


def mutual_information(
    joint: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    joint = joint + epsilon
    p_x = np.sum(joint, axis=1)
    p_y = np.sum(joint, axis=0)

    joint = joint / np.sum(joint)
    p_x = p_x / np.sum(p_x)
    p_y = p_y / np.sum(p_y)

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > epsilon:
                mi += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))

    return mi


def compute_ppmi(
    cooccurrence: np.ndarray,
    smooth: float = 1.0,
    epsilon: float = 1e-10,
) -> np.ndarray:
    total = np.sum(cooccurrence) + smooth * cooccurrence.shape[0] * cooccurrence.shape[1]
    p_ij = (cooccurrence + smooth) / total
    p_i = np.sum(cooccurrence + smooth, axis=1) / total
    p_j = np.sum(cooccurrence + smooth, axis=0) / total

    pmi = np.log2(p_ij / (np.outer(p_i, p_j) + epsilon))
    ppmi = np.maximum(0, pmi)

    return ppmi
