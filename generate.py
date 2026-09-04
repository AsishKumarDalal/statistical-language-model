"""
Load and use the trained statistical language model with GPU support.

Usage:
    python generate.py
    python generate.py --prompt "robin is"
    python generate.py --prompt "robin is" --temperature 0.7
    python generate.py --interactive
"""

import numpy as np
import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_gpu_info, get_gpu_memory_usage, get_device_info, xp, to_device, to_cpu
from training.data import DataManager
from model.embeddings import SVDEmbeddings
from model.sequence import KneserNeyMarkovChain
from model.hmm import HiddenMarkovModel
from model.topics import LDATopicModel
from model.interpolation import ModelInterpolator


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


def load_models(save_dir="saved_models", device="cpu"):
    """Load all trained models."""
    print("\nLoading models...")
    print(f"Device: {device.upper()}")

    data_manager = DataManager()
    data_manager.load_vocabulary(os.path.join(save_dir, "vocabulary.txt"))

    embeddings = SVDEmbeddings(device=device)
    embeddings.load(os.path.join(save_dir, "embeddings.npz"))

    markov = KneserNeyMarkovChain()
    markov.load(os.path.join(save_dir, "markov.pkl"))

    hmm = HiddenMarkovModel()
    hmm.load(os.path.join(save_dir, "hmm.npz"))

    lda = LDATopicModel()
    lda.load(os.path.join(save_dir, "lda.npz"))

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

            xp_ = xp(self.device)
            ctx_emb = to_device(np.mean(self.emb.embeddings[ctx_idx], axis=0), self.device)
            t_idx = self.w2i.get(target, 0)
            if t_idx >= len(self.emb.embeddings): return 1e-10
            t_emb = to_device(self.emb.embeddings[t_idx], self.device)

            if self.device == "cuda":
                sim = float(to_cpu(xp_.dot(ctx_emb, t_emb) / (xp_.linalg.norm(ctx_emb) * xp_.linalg.norm(t_emb) + 1e-10), self.device))
            else:
                sim = float(np.dot(ctx_emb, t_emb) / (np.linalg.norm(ctx_emb) * np.linalg.norm(t_emb) + 1e-10))
            return max(sim, 1e-10)

    interpolator = ModelInterpolator(n_models=4, method="uniform")
    interpolator.add_model(markov, "markov")
    interpolator.add_model(HMMWrapper(hmm, data_manager.idx2word), "hmm")
    interpolator.add_model(LDAWrapper(lda, data_manager.word2idx), "topic")
    interpolator.add_model(EmbWrapper(embeddings, data_manager.word2idx, device), "svd")

    try:
        data = np.load(os.path.join(save_dir, "interpolator.npz"), allow_pickle=True)
        interpolator.weights = data["weights"]
    except:
        interpolator.weights = np.ones(4) / 4

    print("Models loaded successfully!")
    return data_manager, embeddings, markov, hmm, lda, interpolator


def generate_text(
    interpolator,
    data_manager,
    prompt="the",
    max_length=20,
    temperature=0.8,
    top_k=10,
    device="cpu",
):
    """Generate text from prompt with GPU acceleration."""
    print(f"\n{'='*60}")
    print(f"TEXT GENERATION")
    print(f"{'='*60}")

    print(f"Prompt: '{prompt}'")
    print(f"Device: {device.upper()}")
    print(f"Temperature: {temperature}")
    print(f"Max length: {max_length}")
    print("-" * 60)

    context = prompt.split()

    start_time = time.time()
    generated = interpolator.generate(
        context,
        list(data_manager.word2idx.keys()),
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )
    elapsed = time.time() - start_time

    print(f"Generated: {' '.join(generated)}")
    print(f"Generation time: {elapsed*1000:.2f}ms")

    gpu_mem = get_gpu_memory_usage()
    if device == "cuda" and gpu_mem["used"] > 0:
        print(f"GPU Memory: {gpu_mem['used']:.1f} MB")

    print(f"{'='*60}\n")
    return generated


def interactive_mode(interpolator, data_manager, device="cpu"):
    """Interactive text generation mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TEXT GENERATION")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    print("Commands:")
    print("  'quit' - Exit")
    print("  'temp X' - Set temperature (e.g., 'temp 0.7')")
    print("  'len X' - Set max length (e.g., 'len 30')")
    print("  'gpu' - Show GPU status")
    print("  Anything else - Generate text from prompt")
    print("=" * 60)

    temperature = 0.8
    max_length = 20

    while True:
        try:
            user_input = input("\nEnter prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input.lower() == 'quit':
            break
        elif user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature")
            continue
        elif user_input.lower().startswith('len '):
            try:
                max_length = int(user_input.split()[1])
                print(f"Max length set to {max_length}")
            except:
                print("Invalid length")
            continue
        elif user_input.lower() == 'gpu':
            print_gpu_status(device, "Current")
            continue
        elif not user_input:
            user_input = "the"

        generate_text(
            interpolator,
            data_manager,
            prompt=user_input,
            max_length=max_length,
            temperature=temperature,
            device=device,
        )


def show_model_info(data_manager, embeddings, hmm, lda, device):
    """Show information about loaded models."""
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    print(f"Device: {device.upper()}")

    gpu_info = get_gpu_info()
    if device == "cuda" and gpu_info["available"]:
        print(f"GPU: {gpu_info['name']}")
        print(f"GPU Memory: {gpu_info['memory_free']:.0f}MB free / {gpu_info['memory_total']:.0f}MB total")

    print(f"\nVocabulary size: {len(data_manager.word2idx)}")
    print(f"Embedding dimension: {embeddings.embeddings.shape[1]}")
    print(f"HMM states: {hmm.n_states}")
    print(f"LDA topics: {lda.n_topics}")

    print("\nSample word similarities:")
    for word in ["cat", "dog", "the", "good"]:
        if word in embeddings.word2idx:
            similar = embeddings.get_similar_words(word, top_k=3)
            print(f"  '{word}' -> {', '.join([f'{w}({s:.2f})' for w, s in similar])}")

    print("\nLDA Topics:")
    for i in range(min(3, lda.n_topics)):
        words = lda.get_topic_words_with_vocab(i, data_manager.idx2word, top_k=5)
        print(f"  Topic {i}: {', '.join([w for w, _ in words])}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate text with statistical language model")
    parser.add_argument("--prompt", type=str, default="robin is", help="Starting prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum generation length")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Model save directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                        help="Dataset used for training: 'w' for WikiText-2, 'sample', or path")

    args = parser.parse_args()

    # Detect device
    if args.cpu:
        device = "cpu"
    else:
        config = Config()
        device = config.device

    # Show dataset info
    if args.dataset:
        print(f"Trained on dataset: {args.dataset}")
    else:
        print("Dataset: unknown (model files loaded from saved_models/)")

    print("\n" + "=" * 60)
    print("STATISTICAL LANGUAGE MODEL - INFERENCE")
    print("=" * 60)
    print_gpu_status(device, "Startup")

    # Load models
    data_manager, embeddings, markov, hmm, lda, interpolator = load_models(args.save_dir, device)

    # Show model info
    show_model_info(data_manager, embeddings, hmm, lda, device)

    # Generate or interactive
    if args.interactive:
        interactive_mode(interpolator, data_manager, device)
    else:
        generate_text(
            interpolator,
            data_manager,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )


if __name__ == "__main__":
    main()
