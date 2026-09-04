"""
Hidden Markov Model with Baum-Welch (EM) training.
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats import softmax


class HiddenMarkovModel:
    def __init__(
        self,
        n_states: int = 100,
        vocab_size: int = 50000,
        n_iterations: int = 50,
        tolerance: float = 1e-4,
    ):
        self.n_states = n_states
        self.vocab_size = vocab_size
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        self.transition = None
        self.emission = None
        self.initial = None
        self.log_likelihoods = []

    def _initialize_random(self, seed: int = 42):
        np.random.seed(seed)
        self.transition = np.random.dirichlet(
            np.ones(self.n_states) * 0.5, size=self.n_states
        )
        self.emission = np.random.dirichlet(
            np.ones(self.vocab_size) * 0.1, size=self.n_states
        )
        self.initial = np.ones(self.n_states) / self.n_states

    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.initial * self.emission[:, observations[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transition[:, j]) * \
                              self.emission[j, observations[t]]

        log_likelihood = 0.0
        for t in range(T):
            scale = np.sum(alpha[t])
            if scale > 0:
                alpha[t] /= scale
                log_likelihood += np.log(scale)

        return alpha, log_likelihood

    def _backward(self, observations: np.ndarray) -> np.ndarray:
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition[i] * \
                    self.emission[:, observations[t + 1]] * \
                    beta[t + 1]
                )

        for t in range(T - 2, -1, -1):
            scale = np.sum(beta[t])
            if scale > 0:
                beta[t] /= scale

        return beta

    def _forward_backward(
        self,
        observations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        T = len(observations)
        alpha, log_likelihood = self._forward(observations)
        beta = self._backward(observations)

        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-10)
        gamma = gamma / gamma_sum

        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = alpha[t, i] * \
                                   self.transition[i, j] * \
                                   self.emission[j, observations[t + 1]] * \
                                   beta[t + 1, j]

            xi_sum = np.sum(xi[t])
            if xi_sum > 0:
                xi[t] /= xi_sum

        return gamma, xi, alpha, log_likelihood

    def _e_step(
        self,
        sequences: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pi_counts = np.zeros(self.n_states)
        trans_counts = np.zeros((self.n_states, self.n_states))
        emit_counts = np.zeros((self.n_states, self.vocab_size))

        total_log_likelihood = 0.0

        for obs in sequences:
            gamma, xi, _, log_likelihood = self._forward_backward(obs)
            total_log_likelihood += log_likelihood

            pi_counts += gamma[0]
            for t in range(len(obs) - 1):
                trans_counts += xi[t]
            for t in range(len(obs)):
                emit_counts[:, obs[t]] += gamma[t]

        return pi_counts, trans_counts, emit_counts, total_log_likelihood

    def _m_step(
        self,
        pi_counts: np.ndarray,
        trans_counts: np.ndarray,
        emit_counts: np.ndarray,
    ):
        pi_sum = np.sum(pi_counts)
        if pi_sum > 0:
            self.initial = pi_counts / pi_sum

        trans_sums = np.sum(trans_counts, axis=1, keepdims=True)
        trans_sums = np.maximum(trans_sums, 1e-10)
        self.transition = trans_counts / trans_sums

        emit_sums = np.sum(emit_counts, axis=1, keepdims=True)
        emit_sums = np.maximum(emit_sums, 1e-10)
        self.emission = emit_counts / emit_sums

    def fit(
        self,
        sequences: List[np.ndarray],
        verbose: bool = True,
    ):
        self._initialize_random()
        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iterations):
            pi_counts, trans_counts, emit_counts, log_likelihood = \
                self._e_step(sequences)
            self._m_step(pi_counts, trans_counts, emit_counts)
            self.log_likelihoods.append(log_likelihood)

            if verbose:
                perplexity = np.exp(-log_likelihood / sum(len(s) for s in sequences))
                print(f"Iteration {iteration + 1}: "
                      f"LL = {log_likelihood:.2f}, "
                      f"Perplexity = {perplexity:.2f}")

            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            prev_log_likelihood = log_likelihood

    def _viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log(self.initial + 1e-10) + \
                    np.log(self.emission[:, observations[0]] + 1e-10)

        for t in range(1, T):
            for j in range(self.n_states):
                candidates = delta[t - 1] + np.log(self.transition[:, j] + 1e-10)
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + \
                              np.log(self.emission[j, observations[t]] + 1e-10)

        log_prob = np.max(delta[T - 1])
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states, log_prob

    def predict_states(self, observations: np.ndarray) -> np.ndarray:
        states, _ = self._viterbi(observations)
        return states

    def get_state_distribution(self, observations: np.ndarray) -> np.ndarray:
        gamma, _, _, _ = self._forward_backward(observations)
        return gamma

    def predict_next_token(
        self,
        observations: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        gamma = self.get_state_distribution(observations)
        state_dist = gamma[-1]
        token_probs = state_dist @ self.emission
        top_indices = np.argsort(token_probs)[::-1][:top_k]
        return [(idx, float(token_probs[idx])) for idx in top_indices]

    def perplexity(self, sequences: List[np.ndarray]) -> float:
        total_log_prob = 0.0
        total_tokens = 0

        for obs in sequences:
            _, log_likelihood = self._forward(obs)
            total_log_prob += log_likelihood
            total_tokens += len(obs)

        avg_log_prob = total_log_prob / max(total_tokens, 1)
        return np.exp(-avg_log_prob)

    def save(self, path: str):
        np.savez(
            path,
            transition=self.transition,
            emission=self.emission,
            initial=self.initial,
        )
        print(f"Saved HMM to {path}")

    def load(self, path: str):
        data = np.load(path)
        self.transition = data["transition"]
        self.emission = data["emission"]
        self.initial = data["initial"]
        self.n_states = self.transition.shape[0]
        self.vocab_size = self.emission.shape[1]
        print(f"Loaded HMM from {path}")
