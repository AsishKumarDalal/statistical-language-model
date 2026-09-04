"""
Data loading and preprocessing for the statistical language model.
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DataManager:
    def __init__(
        self,
        vocab_size: int = 50000,
        min_token_freq: int = 5,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.min_token_freq = min_token_freq
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.word2idx = {}
        self.idx2word = {}
        self.token_counts = Counter()

        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"
        self.pad = "<PAD>"

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def build_vocabulary(self, texts: List[str]) -> dict:
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)

        self.token_counts = Counter(all_tokens)
        special_tokens = [self.bos, self.eos, self.unk, self.pad]

        filtered = [
            (word, count)
            for word, count in self.token_counts.most_common()
            if count >= self.min_token_freq
        ][:self.vocab_size - len(special_tokens)]

        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}

        for word, _ in filtered:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Unique tokens: {len(self.token_counts)}")

        return self.word2idx

    def text_to_indices(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.word2idx.get(t, self.word2idx[self.unk]) for t in tokens]

    def indices_to_text(self, indices: List[int]) -> str:
        tokens = [self.idx2word.get(idx, self.unk) for idx in indices]
        return " ".join(tokens)

    def prepare_sequences(
        self,
        texts: List[str],
    ) -> List[List[int]]:
        sequences = []
        for text in texts:
            indices = self.text_to_indices(text)
            if len(indices) > 0:
                sequences.append(indices)

        return sequences

    def split_data(
        self,
        sequences: List[List[int]],
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        np.random.shuffle(sequences)

        n = len(sequences)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)

        train = sequences[:n_train]
        val = sequences[n_train:n_train + n_val]
        test = sequences[n_train + n_val:]

        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        return train, val, test

    def extract_ngrams(
        self,
        sequence: List[int],
        order: int,
    ) -> List[Tuple[Tuple[int, ...], int]]:
        ngrams = []
        for i in range(order, len(sequence)):
            context = tuple(sequence[i - order:i])
            target = sequence[i]
            ngrams.append((context, target))

        return ngrams

    def extract_all_ngrams(
        self,
        sequences: List[List[int]],
        max_order: int = 5,
    ) -> dict:
        ngrams_by_order = {}

        for order in range(1, max_order + 1):
            all_ngrams = []
            for seq in sequences:
                ngrams = self.extract_ngrams(seq, order)
                all_ngrams.extend(ngrams)
            ngrams_by_order[order] = all_ngrams

        return ngrams_by_order

    def prepare_validation_data(
        self,
        sequences: List[List[int]],
        max_order: int = 5,
    ) -> List[Tuple[Tuple[int, ...], int]]:
        data = []
        for seq in sequences:
            for order in range(1, max_order + 1):
                ngrams = self.extract_ngrams(seq, order)
                data.extend(ngrams)

        return data

    def create_batches(
        self,
        sequences: List[List[int]],
        batch_size: int = 1024,
    ) -> List[List[List[int]]]:
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batches.append(batch)

        return batches

    def load_text_file(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def load_corpus(
        self,
        filepaths: List[str],
        max_chars: Optional[int] = None,
    ) -> List[str]:
        all_text = []
        total_chars = 0

        for filepath in filepaths:
            text = self.load_text_file(filepath)

            if max_chars is not None:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                text = text[:remaining]
                total_chars += len(text)

            sentences = re.split(r'[.!?]+', text)
            all_text.extend(sentences)

        print(f"Loaded {len(all_text)} sentences")
        return all_text

    def save_vocabulary(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            for word, idx in sorted(self.word2idx.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")
        print(f"Saved vocabulary to {filepath}")

    def load_vocabulary(self, filepath: str):
        self.word2idx = {}
        self.idx2word = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word, idx = line.strip().split('\t')
                idx = int(idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Loaded vocabulary with {len(self.word2idx)} words")
