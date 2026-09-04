"""
Evaluation metrics for the language model.
"""

import numpy as np
from typing import List, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_cross_entropy(
    model,
    test_data: List[Tuple[Tuple[str, ...], str]],
) -> float:
    total_log_prob = 0.0

    for context, target in test_data:
        log_prob = model.get_log_probability(context, target)
        total_log_prob += log_prob

    return -total_log_prob / len(test_data)


def compute_perplexity(
    model,
    test_data: List[Tuple[Tuple[str, ...], str]],
) -> float:
    cross_entropy = compute_cross_entropy(model, test_data)
    return np.exp(cross_entropy)


def compute_token_perplexity(
    model,
    sequences: List[List[int]],
    idx2word: dict,
) -> float:
    total_log_prob = 0.0
    total_tokens = 0

    for seq in sequences:
        for i in range(1, len(seq)):
            context = tuple(idx2word.get(idx, f"<{idx}>") for idx in seq[max(0, i - 5):i])
            target = idx2word.get(seq[i], f"<{seq[i]}>")

            log_prob = model.get_log_probability(context, target)
            total_log_prob += log_prob
            total_tokens += 1

    avg_log_prob = total_log_prob / max(total_tokens, 1)
    return np.exp(-avg_log_prob)


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    p = p / p.sum()
    q = q / q.sum()

    return np.sum(p * np.log(p / q))


def compute_top_k_accuracy(
    model,
    test_data: List[Tuple[Tuple[str, ...], str]],
    k: int = 10,
) -> float:
    correct = 0
    total = len(test_data)

    for context, target in test_data:
        predictions = model.predict_next(list(context), top_k=k)
        predicted_words = [word for word, _ in predictions]

        if target in predicted_words:
            correct += 1

    return correct / total


def compute_entropy(
    probs: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    probs = np.clip(probs, epsilon, 1.0)
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs))


def compute_attention_entropy(
    attention_weights: np.ndarray,
) -> float:
    entropies = []
    for weights in attention_weights:
        entropies.append(compute_entropy(weights))

    return np.mean(entropies)


def evaluate_model(
    model,
    test_sequences: List[List[int]],
    idx2word: dict,
    max_context: int = 5,
) -> dict:
    test_data = []
    for seq in test_sequences:
        for order in range(1, max_context + 1):
            for i in range(order, len(seq)):
                context = tuple(idx2word.get(idx, f"<{idx}>") for idx in seq[i - order:i])
                target = idx2word.get(seq[i], f"<{seq[i]}>")
                test_data.append((context, target))

    cross_entropy = compute_cross_entropy(model, test_data)
    perplexity = compute_perplexity(model, test_data)
    token_perplexity = compute_token_perplexity(model, test_sequences, idx2word)

    metrics = {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "token_perplexity": token_perplexity,
    }

    return metrics


def print_evaluation(metrics: dict):
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Cross-Entropy: {metrics['cross_entropy']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Token Perplexity: {metrics['token_perplexity']:.2f}")
    print("=" * 50)
