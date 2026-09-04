"""
Model Interpolation for combining component models.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats import softmax, cross_entropy


class ModelInterpolator:
    def __init__(self, n_models: int = 4, method: str = "em"):
        self.n_models = n_models
        self.method = method
        self.weights = np.ones(n_models) / n_models
        self.models = []
        self.model_names = ["markov", "hmm", "topic", "svd"]

    def add_model(self, model: Any, name: Optional[str] = None):
        self.models.append(model)
        if name:
            self.model_names.append(name)

    def get_model_probabilities(
        self,
        context: Tuple[str, ...],
        target: str,
    ) -> np.ndarray:
        probs = []
        for model in self.models:
            try:
                prob = model.get_probability(context, target)
                probs.append(prob)
            except Exception:
                probs.append(1e-10)

        return np.array(probs)

    def get_probability(
        self,
        context: Tuple[str, ...],
        target: str,
    ) -> float:
        probs = self.get_model_probabilities(context, target)
        return np.sum(self.weights * probs)

    def get_log_probability(
        self,
        context: Tuple[str, ...],
        target: str,
    ) -> float:
        return np.log(self.get_probability(context, target) + 1e-10)

    def learn_weights_em(
        self,
        validation_data: List[Tuple[Tuple[str, ...], str]],
        n_iterations: int = 20,
        verbose: bool = True,
    ):
        n_models = len(self.models)
        weights = np.ones(n_models) / n_models

        for iteration in range(n_iterations):
            expected = np.zeros(n_models)

            for context, target in validation_data:
                probs = self.get_model_probabilities(context, target)
                posterior = weights * probs
                total = posterior.sum()
                if total > 0:
                    posterior = posterior / total
                    expected += posterior

            if expected.sum() > 0:
                weights = expected / expected.sum()

            if verbose:
                ll = sum(
                    np.log(np.sum(weights * self.get_model_probabilities(ctx, tgt)) + 1e-10)
                    for ctx, tgt in validation_data
                )
                perplexity = np.exp(-ll / len(validation_data))
                print(f"EM iteration {iteration + 1}: "
                      f"Perplexity = {perplexity:.2f}, "
                      f"Weights = {weights}")

        self.weights = weights
        print(f"Final weights: {dict(zip(self.model_names, self.weights))}")

    def learn_weights_grid(
        self,
        validation_data: List[Tuple[Tuple[str, ...], str]],
        resolution: int = 10,
    ):
        best_weights = None
        best_perplexity = np.inf

        for w in np.linspace(0, 1, resolution):
            weights = np.array([w, 1 - w])
            ll = 0.0
            for context, target in validation_data:
                probs = self.get_model_probabilities(context, target)
                ll += np.log(np.sum(weights * probs) + 1e-10)

            perplexity = np.exp(-ll / len(validation_data))
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_weights = weights

        self.weights = best_weights
        print(f"Best weights: {self.weights}, perplexity: {best_perplexity}")

    def learn_weights(
        self,
        validation_data: List[Tuple[Tuple[str, ...], str]],
        **kwargs,
    ):
        if self.method == "em":
            self.learn_weights_em(validation_data, **kwargs)
        elif self.method == "grid":
            self.learn_weights_grid(validation_data, **kwargs)
        elif self.method == "uniform":
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict_next(
        self,
        context: List[str],
        vocab: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        probs = []
        for word in vocab:
            prob = self.get_probability(tuple(context), word)
            probs.append((word, prob))

        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

    def perplexity(
        self,
        test_data: List[Tuple[Tuple[str, ...], str]],
    ) -> float:
        total_log_prob = 0.0

        for context, target in test_data:
            log_prob = self.get_log_probability(context, target)
            total_log_prob += log_prob

        avg_log_prob = total_log_prob / len(test_data)
        return np.exp(-avg_log_prob)

    def generate(
        self,
        context: List[str],
        vocab: List[str],
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[str]:
        generated = list(context)

        for _ in range(max_length):
            predictions = self.predict_next(generated, vocab, top_k=len(vocab))
            words, probs_arr = zip(*predictions)
            probs_arr = np.array(probs_arr)
            probs_arr = np.power(probs_arr + 1e-10, 1.0 / temperature)

            if top_k is not None:
                top_indices = np.argsort(probs_arr)[::-1][:top_k]
                words = [words[i] for i in top_indices]
                probs_arr = probs_arr[top_indices]

            probs_arr = probs_arr / probs_arr.sum()

            idx = np.random.choice(len(words), p=probs_arr)
            next_word = words[idx]

            if next_word == "<EOS>":
                break

            generated.append(next_word)

        return generated

    def save(self, path: str):
        np.savez(
            path,
            weights=self.weights,
            model_names=self.model_names,
        )
        print(f"Saved interpolator to {path}")

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.weights = data["weights"]
        self.model_names = data["model_names"].tolist()
        print(f"Loaded interpolator from {path}")
