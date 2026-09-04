"""
Main training script for the statistical language model.

Usage:
    python train.py                    # Use sample data
    python train.py -d w               # Train on WikiText-2
    python train.py -d w --small       # Use small subset of WikiText-2
    python train.py -d /path/to.txt    # Train on custom text file
"""

import numpy as np
import os
import sys
import time
import argparse
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from config import Config, get_gpu_info, get_gpu_memory_usage
from training.data import DataManager
from model.embeddings import SVDEmbeddings
from model.sequence import KneserNeyMarkovChain
from model.hmm import HiddenMarkovModel
from model.topics import LDATopicModel
from model.interpolation import ModelInterpolator


def download_wikitext2(data_dir="data"):
    """Download WikiText-2 dataset."""
    os.makedirs(data_dir, exist_ok=True)

    alt_urls = {
        "train": "https://raw.githubusercontent.com/eloenter/wikitext-2/master/wiki.train.tokens",
        "valid": "https://raw.githubusercontent.com/eloenter/wikitext-2/master/wiki.valid.tokens",
        "test": "https://raw.githubusercontent.com/eloenter/wikitext-2/master/wiki.test.tokens",
    }

    files = {}
    for split, url in alt_urls.items():
        filepath = os.path.join(data_dir, f"wiki.{split}.tokens")
        if not os.path.exists(filepath):
            print(f"Downloading WikiText-2 {split}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  Downloaded to {filepath}")
            except Exception as e:
                print(f"  Error downloading: {e}")
                print(f"  Please download manually from: https://blog.einstein.ai/the-wikitext-dependency-language-modeling-dataset/")
                return None
        else:
            print(f"  Using existing {filepath}")
        files[split] = filepath

    return files


def load_wikitext2(data_dir="data", small=False):
    """Load WikiText-2 dataset."""
    print("\n" + "=" * 60)
    print("LOADING WIKITEXT-2 DATASET")
    print("=" * 60)

    files = download_wikitext2(data_dir)
    if files is None:
        print("Falling back to sample data...")
        return load_sample_data()

    all_sentences = []
    for split in ["train", "valid", "test"]:
        filepath = files[split]
        print(f"\nLoading {split} from {filepath}...")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = []
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("="):
                parts = line.split(". ")
                for part in parts:
                    part = part.strip()
                    if part and len(part.split()) >= 3:
                        sentences.append(part.lower())

        all_sentences.extend(sentences)
        print(f"  Loaded {len(sentences)} sentences from {split}")

    if small:
        n = len(all_sentences) // 10
        all_sentences = all_sentences[:n]
        print(f"\nUsing small subset: {len(all_sentences)} sentences")
    else:
        print(f"\nTotal sentences: {len(all_sentences)}")

    total_words = sum(len(s.split()) for s in all_sentences)
    print(f"Total words: ~{total_words:,}")
    print(f"Vocabulary size (estimated): ~{len(set(' '.join(all_sentences).split())):,}")

    return all_sentences


def load_sample_data():
    """Load sample data for testing."""
    print("\nUsing sample data (for testing only)")
    sample_sentences = [
        "the cat sat on the mat",
        "the dog played in the yard",
        "the cat chased the mouse",
        "the dog barked at the cat",
        "the sun shone brightly in the sky",
        "the rain fell softly on the ground",
        "the birds sang in the morning",
        "the trees swayed in the wind",
        "the children played in the park",
        "the teachers taught in the school",
        "robin is a bird that flies",
        "robin is used for singing in spring",
        "the robin built a nest in the tree",
        "robin is a common bird name",
        "robin is found in many gardens",
    ] * 200
    return sample_sentences


def load_custom_data(filepath):
    """Load custom text file."""
    print(f"\nLoading custom data from {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line.split()) >= 3:
            sentences.append(line.lower())

    print(f"Loaded {len(sentences)} sentences")
    return sentences


def print_gpu_status(device, phase=""):
    """Print GPU status."""
    if device == "cuda":
        gpu_info = get_gpu_info()
        gpu_mem = get_gpu_memory_usage()
        print(f"\n{'─'*60}")
        print(f"  GPU Status [{phase}]")
        print(f"  Device: {gpu_info['name']}")
        print(f"  Memory: {gpu_mem['used']:.1f} MB used / {gpu_mem['total']:.0f} MB total")
        print(f"{'─'*60}")
    else:
        print(f"\n{'─'*60}")
        print(f"  Device: CPU [{phase}]")
        print(f"{'─'*60}")


def demo_generation(model, data_manager, prompt="the", max_length=20, temperature=0.8):
    print(f"\n{'='*60}")
    print(f"DEMO - Generating text from: '{prompt}'")
    print(f"{'='*60}")

    context = prompt.split()
    try:
        generated = model.generate(
            context,
            list(data_manager.word2idx.keys()),
            max_length=max_length,
            temperature=temperature,
            top_k=10,
        )
        print(f"Generated: {' '.join(generated)}")
    except Exception as e:
        print(f"Generation error: {e}")

    print(f"{'='*60}\n")


def demo_embeddings(embeddings, data_manager):
    print(f"\n{'='*60}")
    print("DEMO - Word Similarities")
    print(f"{'='*60}")

    for word in ["cat", "dog", "the", "good", "king", "queen"]:
        if word in embeddings.word2idx:
            similar = embeddings.get_similar_words(word, top_k=3)
            print(f"'{word}' -> {[(w, f'{s:.3f}') for w, s in similar]}")

    print(f"{'='*60}\n")


def demo_topics(lda, data_manager, n_topics=5):
    print(f"\n{'='*60}")
    print("DEMO - Discovered Topics")
    print(f"{'='*60}")

    for i in range(min(n_topics, lda.n_topics)):
        words = lda.get_topic_words_with_vocab(i, data_manager.idx2word, top_k=8)
        print(f"\nTopic {i}:")
        for w, p in words:
            print(f"  {w}: {p:.4f}")

    print(f"\n{'='*60}\n")


def train(dataset="sample", data_path=None, small=False):
    print("=" * 60)
    print("STATISTICAL LANGUAGE MODEL TRAINING")
    print("=" * 60)

    config = Config()
    config.print_device_info()

    if dataset == "w":
        sentences = load_wikitext2(data_dir="data", small=small)
    elif dataset == "sample":
        sentences = load_sample_data()
    elif os.path.exists(dataset):
        sentences = load_custom_data(dataset)
    else:
        print(f"Unknown dataset: {dataset}")
        print("Options: 'w' for WikiText-2, 'sample' for test data, or path to text file")
        return

    data_manager = DataManager(vocab_size=config.vocab_size, min_token_freq=config.min_token_freq)

    print("\nBuilding vocabulary...")
    data_manager.build_vocabulary(sentences)

    print("\nPreparing sequences...")
    sequences = data_manager.prepare_sequences(sentences)
    train_sequences, val_sequences, test_sequences = data_manager.split_data(sequences)

    total_start = time.time()

    # Phase 1: Embeddings
    print(f"\n{'='*60}")
    print("PHASE 1: SVD Embeddings")
    print(f"{'='*60}")
    print_gpu_status(config.device, "Phase 1")

    start_time = time.time()
    embeddings = SVDEmbeddings(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        window_size=config.cooccurrence_window,
        device=config.device,
    )

    all_tokens = []
    for seq in tqdm(train_sequences, desc="Preparing tokens"):
        tokens = [data_manager.idx2word.get(idx, "<UNK>") for idx in seq]
        all_tokens.extend(tokens)

    embeddings.fit(all_tokens)
    elapsed = time.time() - start_time
    print(f"\nEmbeddings training time: {elapsed:.2f}s")

    demo_embeddings(embeddings, data_manager)

    # Phase 2: Markov Chain
    print(f"\n{'='*60}")
    print("PHASE 2: Markov Chain (Kneser-Ney)")
    print(f"{'='*60}")
    print_gpu_status("cpu", "Phase 2 (CPU-only)")

    start_time = time.time()
    markov = KneserNeyMarkovChain(
        max_order=config.max_ngram_order,
        discount=config.kneser_ney_discount,
        vocab_size=config.vocab_size,
    )

    word_sequences = []
    for seq in tqdm(train_sequences, desc="Converting to words"):
        words = [data_manager.idx2word.get(idx, "<UNK>") for idx in seq]
        word_sequences.append(words)

    print("\nCounting n-grams...")
    for order in range(markov.max_order + 1):
        for seq in tqdm(word_sequences, desc=f"Order {order}"):
            ngrams = markov._get_ngrams(seq, order)
            for context, target in ngrams:
                markov.counts[order][context][target] += 1

    print("\nComputing continuation probabilities...")
    markov._compute_continuation_probs(word_sequences)

    print("\nLearning interpolation weights...")
    markov._learn_interpolation_weights(word_sequences)

    elapsed = time.time() - start_time
    print(f"\nMarkov chain training time: {elapsed:.2f}s")

    demo_generation(markov, data_manager, prompt="robin is")

    # Phase 3: HMM
    print(f"\n{'='*60}")
    print("PHASE 3: HMM (Baum-Welch)")
    print(f"{'='*60}")
    print_gpu_status("cpu", "Phase 3 (CPU-only)")

    start_time = time.time()
    hmm = HiddenMarkovModel(
        n_states=config.hmm_n_states,
        vocab_size=config.vocab_size,
        n_iterations=config.hmm_n_iterations,
        tolerance=config.hmm_tolerance,
    )

    np_sequences = [np.array(seq[:100], dtype=int) for seq in tqdm(train_sequences[:1000], desc="Preparing HMM data")]

    hmm._initialize_random()

    prev_ll = -np.inf
    for iteration in tqdm(range(config.hmm_n_iterations), desc="Baum-Welch EM"):
        pi_counts = np.zeros(hmm.n_states)
        trans_counts = np.zeros((hmm.n_states, hmm.n_states))
        emit_counts = np.zeros((hmm.n_states, config.vocab_size))
        total_ll = 0.0

        for obs in np_sequences:
            gamma, xi, _, ll = hmm._forward_backward(obs)
            total_ll += ll
            pi_counts += gamma[0]
            for t in range(len(obs) - 1):
                trans_counts += xi[t]
            for t in range(len(obs)):
                emit_counts[:, obs[t]] += gamma[t]

        hmm._m_step(pi_counts, trans_counts, emit_counts)

        if iteration % 10 == 0:
            pp = np.exp(-total_ll / sum(len(s) for s in np_sequences))
            tqdm.write(f"  Iter {iteration}: LL={total_ll:.2f}, PP={pp:.2f}")

        if abs(total_ll - prev_ll) < config.hmm_tolerance:
            tqdm.write(f"  Converged at iteration {iteration}")
            break
        prev_ll = total_ll

    elapsed = time.time() - start_time
    print(f"\nHMM training time: {elapsed:.2f}s")

    demo_generation(markov, data_manager, prompt="the cat")

    # Phase 4: LDA
    print(f"\n{'='*60}")
    print("PHASE 4: LDA Topic Model (Gibbs Sampling)")
    print(f"{'='*60}")
    print_gpu_status("cpu", "Phase 4 (CPU-only)")

    start_time = time.time()
    lda = LDATopicModel(
        n_topics=config.n_topics,
        alpha=config.lda_alpha,
        beta=config.lda_beta,
        n_iterations=config.lda_n_iterations,
        burn_in=config.lda_burn_in,
    )

    documents = [seq for seq in train_sequences if len(seq) > 0]
    lda._initialize_counts(documents, len(documents), config.vocab_size)

    doc_topic_accum = np.zeros_like(lda.doc_topic_counts)
    topic_word_accum = np.zeros_like(lda.topic_word_counts)

    for iteration in tqdm(range(config.lda_n_iterations), desc="Gibbs sampling"):
        lda._gibbs_sample(documents, len(documents), config.vocab_size)
        if iteration >= config.lda_burn_in:
            doc_topic_accum += lda.doc_topic_counts
            topic_word_accum += lda.topic_word_counts
        if iteration % 100 == 0:
            tqdm.write(f"  Iter {iteration}")

    n_samples = config.lda_n_iterations - config.lda_burn_in
    lda.doc_topic_dist = doc_topic_accum / n_samples
    lda.topic_word_dist = topic_word_accum / n_samples
    lda.doc_topic_dist /= lda.doc_topic_dist.sum(axis=1, keepdims=True) + 1e-10
    lda.topic_word_dist /= lda.topic_word_dist.sum(axis=1, keepdims=True) + 1e-10

    elapsed = time.time() - start_time
    print(f"\nLDA training time: {elapsed:.2f}s")

    demo_topics(lda, data_manager)

    # Phase 5: Interpolation
    print(f"\n{'='*60}")
    print("PHASE 5: Model Interpolation")
    print(f"{'='*60}")
    print_gpu_status(config.device, "Phase 5")

    start_time = time.time()
    interpolator = ModelInterpolator(n_models=4, method="em")

    interpolator.add_model(markov, "markov")

    class HMMWrapper:
        def __init__(self, hmm_model, idx2word):
            self.hmm = hmm_model
            self.idx2word = idx2word
        def get_probability(self, context, target):
            ctx_idx = np.array([self._w2i(w) for w in context], dtype=int)
            preds = self.hmm.predict_next_token(ctx_idx, top_k=100)
            return dict(preds).get(self._w2i(target), 1e-10)
        def _w2i(self, w):
            for i, word in self.idx2word.items():
                if word == w: return i
            return 0

    class LDAWrapper:
        def __init__(self, lda_model, w2i):
            self.lda = lda_model
            self.w2i = w2i
        def get_probability(self, context, target):
            ctx_idx = [self.w2i.get(w, 0) for w in context]
            theta = self.lda.predict_topic_distribution(ctx_idx)
            t_idx = self.w2i.get(target, 0)
            return max(np.sum(self.lda.topic_word_dist[:, t_idx] * theta), 1e-10)

    class EmbWrapper:
        def __init__(self, emb, w2i, device):
            self.emb = emb
            self.w2i = w2i
            self.device = device
        def get_probability(self, context, target):
            if not context: return 1e-10
            ctx_idx = [self.w2i.get(w, 0) for w in context]
            ctx_idx = [i for i in ctx_idx if i < len(self.emb.embeddings)]
            if not ctx_idx: return 1e-10
            ctx_emb = np.mean(self.emb.embeddings[ctx_idx], axis=0)
            t_idx = self.w2i.get(target, 0)
            if t_idx >= len(self.emb.embeddings): return 1e-10
            t_emb = self.emb.embeddings[t_idx]
            sim = np.dot(ctx_emb, t_emb) / (np.linalg.norm(ctx_emb) * np.linalg.norm(t_emb) + 1e-10)
            return max(sim, 1e-10)

    interpolator.add_model(HMMWrapper(hmm, data_manager.idx2word), "hmm")
    interpolator.add_model(LDAWrapper(lda, data_manager.word2idx), "topic")
    interpolator.add_model(EmbWrapper(embeddings, data_manager.word2idx, config.device), "svd")

    val_data = []
    for seq in tqdm(val_sequences, desc="Preparing validation"):
        words = [data_manager.idx2word.get(idx, "<UNK>") for idx in seq]
        for i in range(1, len(words)):
            val_data.append((tuple(words[max(0,i-5):i]), words[i]))

    n_models = 4
    weights = np.ones(n_models) / n_models

    for iteration in tqdm(range(20), desc="EM for interpolation"):
        expected = np.zeros(n_models)
        for context, target in val_data[:1000]:
            probs = np.array([m.get_probability(context, target) for m in interpolator.models])
            posterior = weights * probs
            if posterior.sum() > 0:
                posterior = posterior / posterior.sum()
                expected += posterior
        if expected.sum() > 0:
            weights = expected / expected.sum()

    interpolator.weights = weights
    elapsed = time.time() - start_time
    print(f"\nInterpolation training time: {elapsed:.2f}s")

    print(f"\n{'='*60}")
    print("FINAL DEMOS")
    print(f"{'='*60}")
    print_gpu_status(config.device, "Inference")

    demo_generation(interpolator, data_manager, prompt="robin is")
    demo_generation(interpolator, data_manager, prompt="the cat")
    demo_generation(interpolator, data_manager, prompt="in the")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL TRAINING TIME: {total_elapsed:.2f}s ({total_elapsed/3600:.2f} hours)")
    print(f"Dataset: {dataset}")
    print(f"Device: {config.device.upper()}")
    print(f"{'='*60}")

    print("\nSaving models...")
    os.makedirs("saved_models", exist_ok=True)
    embeddings.save("saved_models/embeddings.npz")
    markov.save("saved_models/markov.pkl")
    hmm.save("saved_models/hmm.npz")
    lda.save("saved_models/lda.npz")
    interpolator.save("saved_models/interpolator.npz")
    data_manager.save_vocabulary("saved_models/vocabulary.txt")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train statistical language model")
    parser.add_argument("-d", "--dataset", type=str, default="sample",
                        help="Dataset: 'w' for WikiText-2, 'sample' for test data, or path to text file")
    parser.add_argument("--small", action="store_true",
                        help="Use small subset of dataset (for quick testing)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode")

    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train(dataset=args.dataset, small=args.small)
