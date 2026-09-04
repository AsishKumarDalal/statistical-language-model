"""
SVD-based word embeddings with GPU support.

Statistical interpretation:
- Co-occurrence matrix captures distributional semantics
- PPMI weighting upweights rare but meaningful pairs
- SVD finds optimal low-rank approximation
- Embeddings = U * sqrt(Sigma) from truncated SVD
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats import compute_ppmi
from config import xp, to_device, to_cpu, get_gpu_memory_usage


class SVDEmbeddings:
    """
    Word embeddings via SVD on PPMI matrix with GPU support.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 300,
        window_size: int = 5,
        ppmi_smooth: float = 1.0,
        min_count: int = 5,
        device: str = "cpu",
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.ppmi_smooth = ppmi_smooth
        self.min_count = min_count
        self.device = device

        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
        self.cooccurrence = None
        self.ppmi = None

    def build_vocabulary(self, tokens: List[str]) -> dict:
        token_counts = Counter(tokens)
        filtered = [
            (word, count)
            for word, count in token_counts.most_common()
            if count >= self.min_count
        ][:self.vocab_size]

        self.word2idx = {word: idx for idx, (word, _) in enumerate(filtered)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Vocabulary size: {len(self.word2idx)}")
        return self.word2idx

    def build_cooccurrence_matrix(self, tokens: List[str]) -> np.ndarray:
        token_indices = [self.word2idx[t] for t in tokens if t in self.word2idx]
        n = len(self.word2idx)

        xp_ = xp(self.device)
        cooccurrence = xp_.zeros((n, n), dtype=xp_.float32)

        print(f"Building co-occurrence matrix on {self.device.upper()}...")

        for i, idx in enumerate(token_indices):
            start = max(0, i - self.window_size)
            end = min(len(token_indices), i + self.window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_idx = token_indices[j]
                    cooccurrence[idx, context_idx] += 1.0

        self.cooccurrence = to_cpu(cooccurrence, self.device)
        print(f"Co-occurrence matrix shape: {cooccurrence.shape}")
        print(f"Non-zero entries: {np.count_nonzero(self.cooccurrence)}")

        return self.cooccurrence

    def compute_ppmi_matrix(self) -> np.ndarray:
        if self.cooccurrence is None:
            raise ValueError("Build co-occurrence matrix first")

        self.ppmi = compute_ppmi(self.cooccurrence, smooth=self.ppmi_smooth)
        print(f"PPMI matrix shape: {self.ppmi.shape}")
        print(f"Mean PPMI: {np.mean(self.ppmi):.4f}")

        return self.ppmi

    def fit(self, tokens: List[str]) -> np.ndarray:
        print(f"\n{'='*60}")
        print(f"Training SVD Embeddings on {self.device.upper()}")
        print(f"{'='*60}")

        self.build_vocabulary(tokens)
        self.build_cooccurrence_matrix(tokens)
        self.compute_ppmi_matrix()

        print(f"Computing SVD on {self.device.upper()}...")

        xp_ = xp(self.device)
        ppmi_device = to_device(self.ppmi, self.device)

        U_device, sigma_device, Vt_device = xp_.linalg.svd(ppmi_device, full_matrices=False)

        U_cpu = to_cpu(U_device, self.device)[:, :self.embed_dim]
        sigma_cpu = to_cpu(sigma_device, self.device)[:self.embed_dim]

        self.embeddings = U_cpu * np.sqrt(sigma_cpu)

        print(f"Embeddings shape: {self.embeddings.shape}")

        gpu_mem = get_gpu_memory_usage()
        if self.device == "cuda" and gpu_mem["used"] > 0:
            print(f"GPU Memory used: {gpu_mem['used']:.1f} MB")

        return self.embeddings

    def get_embedding(self, word: str) -> np.ndarray:
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        return self.embeddings[self.word2idx[word]]

    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")

        query_idx = self.word2idx[word]
        query = self.embeddings[query_idx]

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = self.embeddings / norms

        query_norm = query / (np.linalg.norm(query) + 1e-10)
        similarities = normalized @ query_norm

        top_indices = np.argsort(similarities)[::-1][1:top_k + 1]
        return [(self.idx2word[idx], float(similarities[idx])) for idx in top_indices]

    def most_similar(self, words: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        query_indices = [self.word2idx[w] for w in words if w in self.word2idx]
        if not query_indices:
            raise ValueError("No query words in vocabulary")

        query = np.mean(self.embeddings[query_indices], axis=0)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = self.embeddings / norms

        query_norm = query / (np.linalg.norm(query) + 1e-10)
        similarities = normalized @ query_norm

        query_set = set(query_indices)
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if idx not in query_set:
                results.append((self.idx2word[idx], float(similarities[idx])))
                if len(results) >= top_k:
                    break
        return results

    def save(self, path: str):
        np.savez(path, embeddings=self.embeddings, word2idx=self.word2idx, idx2word=self.idx2word)
        print(f"Saved embeddings to {path}")

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.word2idx = data["word2idx"].item()
        self.idx2word = data["idx2word"].item()
        print(f"Loaded embeddings from {path}")
