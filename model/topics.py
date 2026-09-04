"""
Latent Dirichlet Allocation (LDA) with Gibbs Sampling.
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LDATopicModel:
    def __init__(
        self,
        n_topics: int = 50,
        alpha: float = 0.1,
        beta: float = 0.01,
        n_iterations: int = 500,
        burn_in: int = 100,
        seed: int = 42,
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iterations = n_iterations
        self.burn_in = burn_in

        np.random.seed(seed)

        self.doc_topic_counts = None
        self.topic_word_counts = None
        self.topic_counts = None

        self.doc_topic_dist = None
        self.topic_word_dist = None

        self.word2idx = {}
        self.idx2word = {}

    def _initialize_counts(
        self,
        documents: List[List[int]],
        n_docs: int,
        vocab_size: int,
    ):
        self.doc_topic_counts = np.zeros((n_docs, self.n_topics))
        self.topic_word_counts = np.zeros((self.n_topics, vocab_size))
        self.topic_counts = np.zeros(self.n_topics)

        self.topic_assignments = []
        for doc in documents:
            doc_topics = np.random.randint(0, self.n_topics, len(doc))
            self.topic_assignments.append(doc_topics)

            for word_idx, topic in zip(doc, doc_topics):
                self.doc_topic_counts[len(self.topic_assignments) - 1, topic] += 1
                self.topic_word_counts[topic, word_idx] += 1
                self.topic_counts[topic] += 1

    def _gibbs_sample(
        self,
        documents: List[List[int]],
        n_docs: int,
        vocab_size: int,
    ):
        for doc_idx, doc in enumerate(documents):
            for word_pos, word_idx in enumerate(doc):
                old_topic = self.topic_assignments[doc_idx][word_pos]

                self.doc_topic_counts[doc_idx, old_topic] -= 1
                self.topic_word_counts[old_topic, word_idx] -= 1
                self.topic_counts[old_topic] -= 1

                probs = np.zeros(self.n_topics)
                for k in range(self.n_topics):
                    doc_prob = self.doc_topic_counts[doc_idx, k] + self.alpha
                    word_prob = (self.topic_word_counts[k, word_idx] + self.beta) / \
                                (self.topic_counts[k] + vocab_size * self.beta)
                    probs[k] = doc_prob * word_prob

                probs = probs / probs.sum()
                new_topic = np.random.choice(self.n_topics, p=probs)

                self.topic_assignments[doc_idx][word_pos] = new_topic
                self.doc_topic_counts[doc_idx, new_topic] += 1
                self.topic_word_counts[new_topic, word_idx] += 1
                self.topic_counts[new_topic] += 1

    def fit(
        self,
        documents: List[List[int]],
        vocab_size: int,
        verbose: bool = True,
    ):
        n_docs = len(documents)
        self._initialize_counts(documents, n_docs, vocab_size)

        doc_topic_accum = np.zeros_like(self.doc_topic_counts)
        topic_word_accum = np.zeros_like(self.topic_word_counts)

        for iteration in tqdm(range(self.n_iterations), desc="Gibbs sampling"):
            self._gibbs_sample(documents, n_docs, vocab_size)
            if iteration >= self.burn_in:
                doc_topic_accum += self.doc_topic_counts
                topic_word_accum += self.topic_word_counts

        n_samples = self.n_iterations - self.burn_in
        self.doc_topic_dist = doc_topic_accum / n_samples
        self.topic_word_dist = topic_word_accum / n_samples

        doc_sums = self.doc_topic_dist.sum(axis=1, keepdims=True)
        doc_sums = np.maximum(doc_sums, 1e-10)
        self.doc_topic_dist = self.doc_topic_dist / doc_sums

        topic_sums = self.topic_word_dist.sum(axis=1, keepdims=True)
        topic_sums = np.maximum(topic_sums, 1e-10)
        self.topic_word_dist = self.topic_word_dist / topic_sums

        if verbose:
            print("LDA training complete!")
            print(f"Document-topic shape: {self.doc_topic_dist.shape}")
            print(f"Topic-word shape: {self.topic_word_dist.shape}")

    def get_document_topics(
        self,
        doc_idx: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        probs = self.doc_topic_dist[doc_idx]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(idx, float(probs[idx])) for idx in top_indices]

    def get_topic_words(
        self,
        topic_idx: int,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        probs = self.topic_word_dist[topic_idx]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(idx, float(probs[idx])) for idx in top_indices]

    def get_topic_words_with_vocab(
        self,
        topic_idx: int,
        idx2word: dict,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        word_probs = self.get_topic_words(topic_idx, top_k)
        return [(idx2word.get(idx, f"<{idx}>"), prob) for idx, prob in word_probs]

    def predict_topic_distribution(self, document: List[int]) -> np.ndarray:
        topic_counts = np.zeros(self.n_topics)

        for word_idx in document:
            for k in range(self.n_topics):
                topic_counts[k] += self.topic_word_dist[k, word_idx]

        topic_counts = topic_counts / (topic_counts.sum() + 1e-10)
        return topic_counts

    def perplexity(self, documents: List[List[int]]) -> float:
        total_log_prob = 0.0
        total_words = 0

        for doc_idx, doc in enumerate(documents):
            if doc_idx < len(self.doc_topic_dist):
                theta = self.doc_topic_dist[doc_idx]
            else:
                theta = self.predict_topic_distribution(doc)

            for word_idx in doc:
                p_word = np.sum(self.topic_word_dist[:, word_idx] * theta)
                total_log_prob += np.log(p_word + 1e-10)
                total_words += 1

        avg_log_prob = total_log_prob / max(total_words, 1)
        return np.exp(-avg_log_prob)

    def topic_coherence(
        self,
        documents: List[List[int]],
        top_k: int = 10,
    ) -> float:
        word_doc_count = {}
        for doc in documents:
            for word in set(doc):
                word_doc_count[word] = word_doc_count.get(word, 0) + 1

        total_docs = len(documents)
        coherence_sum = 0.0
        n_topics = 0

        for topic_idx in range(self.n_topics):
            top_words = [idx for idx, _ in self.get_topic_words(topic_idx, top_k)]
            topic_coherence = 0.0
            n_pairs = 0

            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    w1, w2 = top_words[i], top_words[j]

                    count_w1 = word_doc_count.get(w1, 0)
                    count_w2 = word_doc_count.get(w2, 0)

                    co_count = 0
                    for doc in documents:
                        if w1 in doc and w2 in doc:
                            co_count += 1

                    if count_w1 > 0 and count_w2 > 0:
                        pmi = np.log2(
                            (co_count / total_docs + 1e-10) /
                            (count_w1 / total_docs * count_w2 / total_docs + 1e-10)
                        )
                        topic_coherence += pmi
                        n_pairs += 1

            if n_pairs > 0:
                coherence_sum += topic_coherence / n_pairs
                n_topics += 1

        return coherence_sum / max(n_topics, 1)

    def save(self, path: str):
        np.savez(
            path,
            doc_topic_dist=self.doc_topic_dist,
            topic_word_dist=self.topic_word_dist,
        )
        print(f"Saved LDA to {path}")

    def load(self, path: str):
        data = np.load(path)
        self.doc_topic_dist = data["doc_topic_dist"]
        self.topic_word_dist = data["topic_word_dist"]
        self.n_topics = self.topic_word_dist.shape[0]
        print(f"Loaded LDA from {path}")
