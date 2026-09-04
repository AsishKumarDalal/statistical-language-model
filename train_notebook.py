"""
Jupyter Notebook for training the statistical language model.
Copy this into a .ipynb file or run as script.
"""

import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))

try:
    from tqdm.notebook import tqdm
    print("Using tqdm.notebook (Jupyter mode)")
except:
    from tqdm import tqdm
    print("Using tqdm (terminal mode)")

from config import Config
from training.data import DataManager
from model.embeddings import SVDEmbeddings
from model.sequence import KneserNeyMarkovChain
from model.hmm import HiddenMarkovModel
from model.topics import LDATopicModel
from model.interpolation import ModelInterpolator

print("All imports successful!")

config = Config()
print(f"Vocab size: {config.vocab_size}")
print(f"Embedding dim: {config.embed_dim}")
print(f"HMM states: {config.hmm_n_states}")
print(f"Topics: {config.n_topics}")

def load_sample_data():
    sample_sentences = [
        "the cat sat on the mat",
        "the dog played in the yard",
        "the cat chased the mouse",
        "the dog barked at the cat",
        "the sun shone brightly in the sky",
        "the rain fell softly on the ground",
        "robin is a bird that flies",
        "robin is used for singing in spring",
        "the robin built a nest in the tree",
        "robin is a common bird name",
    ] * 200
    return sample_sentences

sentences = load_sample_data()
print(f"Loaded {len(sentences)} sentences")

data_manager = DataManager(vocab_size=config.vocab_size, min_token_freq=config.min_token_freq)
data_manager.build_vocabulary(sentences)
sequences = data_manager.prepare_sequences(sentences)
train_seq, val_seq, test_seq = data_manager.split_data(sequences)

print("\n" + "="*60)
print("PHASE 1: SVD Embeddings")
print("="*60)

embeddings = SVDEmbeddings(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    window_size=config.cooccurrence_window,
)

all_tokens = []
for seq in tqdm(train_seq, desc="Preparing tokens"):
    tokens = [data_manager.idx2word.get(idx, "<UNK>") for idx in seq]
    all_tokens.extend(tokens)

embeddings.build_vocabulary(all_tokens)
embeddings.build_cooccurrence_matrix(all_tokens)
embeddings.compute_ppmi_matrix()

print("Computing SVD...")
U, sigma, Vt = np.linalg.svd(embeddings.ppmi, full_matrices=False)
embeddings.embeddings = U[:, :config.embed_dim] * np.sqrt(sigma[:config.embed_dim])

print(f"Embeddings shape: {embeddings.embeddings.shape}")

print("\n--- DEMO: Word Similarities ---")
for word in ["cat", "dog", "the"]:
    if word in embeddings.word2idx:
        similar = embeddings.get_similar_words(word, top_k=3)
        print(f"'{word}' -> {[(w, f'{s:.3f}') for w, s in similar]}")

print("\n" + "="*60)
print("PHASE 2: Markov Chain (Kneser-Ney)")
print("="*60)

markov = KneserNeyMarkovChain(
    max_order=config.max_ngram_order,
    discount=config.kneser_ney_discount,
    vocab_size=config.vocab_size,
)

word_sequences = []
for seq in tqdm(train_seq, desc="Converting to words"):
    words = [data_manager.idx2word.get(idx, "<UNK>") for idx in seq]
    word_sequences.append(words)

for order in range(markov.max_order + 1):
    for seq in tqdm(word_sequences, desc=f"Order {order}"):
        ngrams = markov._get_ngrams(seq, order)
        for context, target in ngrams:
            markov.counts[order][context][target] += 1

markov._compute_continuation_probs(word_sequences)
markov._learn_interpolation_weights(word_sequences)

print("\n--- DEMO: Markov Chain Generation ---")
gen = markov.generate(["robin", "is"], max_length=10, temperature=0.8)
print(f"Generated: {' '.join(gen)}")

print("\n" + "="*60)
print("PHASE 3: HMM (Baum-Welch)")
print("="*60)

hmm = HiddenMarkovModel(
    n_states=config.hmm_n_states,
    vocab_size=config.vocab_size,
    n_iterations=config.hmm_n_iterations,
    tolerance=config.hmm_tolerance,
)

np_sequences = [np.array(seq[:100], dtype=int) for seq in tqdm(train_seq[:500], desc="Preparing HMM data")]

hmm._initialize_random()

prev_ll = -np.inf
for iteration in tqdm(range(config.hmm_n_iterations), desc="Baum-Welch"):
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
        print(f"Iter {iteration}: LL={total_ll:.2f}, PP={pp:.2f}")

    if abs(total_ll - prev_ll) < config.hmm_tolerance:
        print(f"Converged at iteration {iteration}")
        break
    prev_ll = total_ll

print("\n" + "="*60)
print("PHASE 4: LDA (Gibbs Sampling)")
print("="*60)

lda = LDATopicModel(
    n_topics=config.n_topics,
    alpha=config.lda_alpha,
    beta=config.lda_beta,
    n_iterations=config.lda_n_iterations,
    burn_in=config.lda_burn_in,
)

documents = [seq for seq in train_seq if len(seq) > 0]
lda._initialize_counts(documents, len(documents), config.vocab_size)

doc_topic_accum = np.zeros_like(lda.doc_topic_counts)
topic_word_accum = np.zeros_like(lda.topic_word_counts)

for iteration in tqdm(range(config.lda_n_iterations), desc="Gibbs sampling"):
    lda._gibbs_sample(documents, len(documents), config.vocab_size)
    if iteration >= config.lda_burn_in:
        doc_topic_accum += lda.doc_topic_counts
        topic_word_accum += lda.topic_word_counts

n_samples = config.lda_n_iterations - config.lda_burn_in
lda.doc_topic_dist = doc_topic_accum / n_samples
lda.topic_word_dist = topic_word_accum / n_samples
lda.doc_topic_dist /= lda.doc_topic_dist.sum(axis=1, keepdims=True) + 1e-10
lda.topic_word_dist /= lda.topic_word_dist.sum(axis=1, keepdims=True) + 1e-10

print("\n--- DEMO: Discovered Topics ---")
for i in range(3):
    words = lda.get_topic_words_with_vocab(i, data_manager.idx2word, top_k=5)
    print(f"Topic {i}: {', '.join([w for w, _ in words])}")

print("\n" + "="*60)
print("PHASE 5: Model Interpolation")
print("="*60)

interpolator = ModelInterpolator(n_models=4, method="em")

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
    def __init__(self, emb, w2i):
        self.emb = emb
        self.w2i = w2i
    def get_probability(self, context, target):
        if not context: return 1e-10
        ctx_idx = [self.w2i.get(w, 0) for w in context]
        ctx_idx = [i for i in ctx_idx if i < len(self.emb.embeddings)]
        if not ctx_idx: return 1e-10
        ctx_emb = np.mean(self.emb.embeddings[ctx_idx], axis=0)
        t_idx = self.w2i.get(target, 0)
        if t_idx >= len(self.emb.embeddings): return 1e-10
        t_emb = self.emb.embeddings[t_idx]
        return max(np.dot(ctx_emb, t_emb) / (np.linalg.norm(ctx_emb) * np.linalg.norm(t_emb) + 1e-10), 1e-10)

interpolator.add_model(markov, "markov")
interpolator.add_model(HMMWrapper(hmm, data_manager.idx2word), "hmm")
interpolator.add_model(LDAWrapper(lda, data_manager.word2idx), "topic")
interpolator.add_model(EmbWrapper(embeddings, data_manager.word2idx), "svd")

val_data = []
for seq in tqdm(val_seq, desc="Preparing val data"):
    words = [data_manager.idx2word.get(idx, "<UNK>") for idx in seq]
    for i in range(1, len(words)):
        val_data.append((tuple(words[max(0,i-5):i]), words[i]))

interpolator.learn_weights(val_data[:1000])

print("\n--- DEMO: Final Model Generation ---")
for prompt in ["robin is", "the cat", "in the"]:
    gen = interpolator.generate(prompt.split(), list(data_manager.word2idx.keys()), max_length=15)
    print(f"'{prompt}' -> {' '.join(gen)}")

print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

os.makedirs("saved_models", exist_ok=True)
embeddings.save("saved_models/embeddings.npz")
markov.save("saved_models/markov.pkl")
hmm.save("saved_models/hmm.npz")
lda.save("saved_models/lda.npz")
interpolator.save("saved_models/interpolator.npz")
data_manager.save_vocabulary("saved_models/vocabulary.txt")

print("\nAll models saved!")
print("Run 'python generate.py --prompt \"robin is\"' to use the model")
