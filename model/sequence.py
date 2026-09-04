"""
N-gram Markov Chain with Kneser-Ney Smoothing.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KneserNeyMarkovChain:
    def __init__(
        self,
        max_order: int = 5,
        discount: float = 0.75,
        vocab_size: int = 50000,
    ):
        self.max_order = max_order
        self.discount = discount
        self.vocab_size = vocab_size

        self.counts = {}
        for order in range(max_order + 1):
            self.counts[order] = defaultdict(Counter)

        self.context_totals = {}
        self.lambdas = None

        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"

    def _get_ngrams(
        self,
        tokens: List[str],
        order: int,
    ) -> List[Tuple[Tuple[str, ...], str]]:
        padded = [self.bos] * order + tokens + [self.eos]
        ngrams = []
        for i in range(order, len(padded)):
            context = tuple(padded[i - order:i])
            target = padded[i]
            ngrams.append((context, target))

        return ngrams

    def fit(self, token_sequences: List[List[str]]):
        print("Counting n-grams...")
        for order in range(self.max_order + 1):
            for seq in tqdm(token_sequences, desc=f"Order {order}"):
                ngrams = self._get_ngrams(seq, order)
                for context, target in ngrams:
                    self.counts[order][context][target] += 1

            self.context_totals[order] = {
                ctx: sum(counts.values())
                for ctx, counts in self.counts[order].items()
            }

        print("Computing continuation probabilities for Kneser-Ney...")
        self._compute_continuation_probs(token_sequences)

        print("Learning interpolation weights via EM...")
        self._learn_interpolation_weights(token_sequences)

        print("Markov chain trained!")

    def _compute_continuation_probs(self, token_sequences: List[List[str]]):
        word_contexts = defaultdict(set)

        for order in range(1, self.max_order + 1):
            for seq in token_sequences:
                padded = [self.bos] * order + seq + [self.eos]
                for i in range(order, len(padded)):
                    context = tuple(padded[i - order:i])
                    word = padded[i]
                    word_contexts[word].add((order, context))

        total_contexts = sum(len(ctxs) for ctxs in word_contexts.values())
        self.continuation_prob = {
            word: len(ctxs) / total_contexts
            for word, ctxs in word_contexts.items()
        }

    def _learn_interpolation_weights(self, token_sequences: List[List[str]]):
        n_orders = self.max_order + 1
        lambdas = np.ones(n_orders) / n_orders

        for iteration in range(20):
            expected = np.zeros(n_orders)

            for seq in token_sequences[:1000]:
                ngrams = self._get_ngrams(seq, self.max_order)

                for context, target in ngrams:
                    probs = []
                    for order in range(n_orders):
                        if order <= len(context):
                            ctx = context[-order:] if order > 0 else ()
                            count = self.counts[order][ctx].get(target, 0)
                            total = self.context_totals[order].get(ctx, 0)
                            probs.append(count / max(total, 1))
                        else:
                            probs.append(0)

                    probs = np.array(probs)
                    if probs.sum() > 0:
                        posterior = lambdas * probs
                        posterior = posterior / posterior.sum()
                        expected += posterior

            if expected.sum() > 0:
                lambdas = expected / expected.sum()

        self.lambdas = lambdas
        print(f"Interpolation weights: {self.lambdas}")

    def get_probability(
        self,
        context: Tuple[str, ...],
        target: str,
    ) -> float:
        prob = 0.0

        for order in range(min(len(context) + 1, self.max_order + 1)):
            if order == 0:
                p_order = 1.0 / self.vocab_size
            else:
                ctx = context[-order:]
                count = self.counts[order][ctx].get(target, 0)
                total = self.context_totals[order].get(ctx, 0)

                if total > 0:
                    p_order = max(count - self.discount, 0) / total
                    if count == 0:
                        p_order += self.discount / total * self.continuation_prob.get(target, 1.0 / self.vocab_size)
                else:
                    p_order = 1.0 / self.vocab_size

            prob += self.lambdas[order] * p_order

        return prob

    def get_log_probability(
        self,
        context: Tuple[str, ...],
        target: str,
    ) -> float:
        return np.log(self.get_probability(context, target) + 1e-10)

    def predict_next(
        self,
        context: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        ctx = tuple(context[-self.max_order:])
        probs = []
        for word in self.word2idx:
            prob = self.get_probability(ctx, word)
            probs.append((word, prob))

        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

    def perplexity(self, token_sequences: List[List[str]]) -> float:
        total_log_prob = 0.0
        total_tokens = 0

        for seq in token_sequences:
            ngrams = self._get_ngrams(seq, self.max_order)
            for context, target in ngrams:
                log_prob = self.get_log_probability(context, target)
                total_log_prob += log_prob
                total_tokens += 1

        avg_log_prob = total_log_prob / max(total_tokens, 1)
        return np.exp(-avg_log_prob)

    def generate(
        self,
        context: List[str],
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> List[str]:
        generated = list(context)

        for _ in range(max_length):
            ctx = tuple(generated[-self.max_order:])
            probs = []
            for word in self.word2idx:
                prob = self.get_probability(ctx, word)
                probs.append((word, prob))

            words, probs_arr = zip(*probs)
            probs_arr = np.array(probs_arr)
            probs_arr = np.power(probs_arr, 1.0 / temperature)
            probs_arr = probs_arr / probs_arr.sum()

            idx = np.random.choice(len(words), p=probs_arr)
            next_word = words[idx]

            if next_word == self.eos:
                break

            generated.append(next_word)

        return generated

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved Markov chain to {path}")

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
        print(f"Loaded Markov chain from {path}")
