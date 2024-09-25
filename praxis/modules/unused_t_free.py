import hashlib
import math
import random
import re
import string
from collections import Counter, defaultdict
from itertools import islice

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TFreeConfig(PretrainedConfig):
    model_type = "t_free"

    def __init__(self, vocab_size=8000, embedding_dim=768, m=8, k=3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.m = m
        self.k = k
        self.pad_token_id = 0  # Add this line


class TFreeTokenizer(PreTrainedTokenizer):
    def __init__(self, config):
        self.config = config
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.encoder = {
            self.pad_token: 0,
            self.unk_token: 1,
        }
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
        )

        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.unk_token_id = self.convert_tokens_to_ids(self.unk_token)

        self.m = config.m
        self.k = config.k
        self.hash_function = hashlib.sha256

        # Initialize an empty reverse mapping
        self.trigram_id_to_trigrams = {}

        # Initialize trigram frequencies
        self.trigram_frequencies = Counter()

    def encode(self, text, **kwargs):
        tokens = self._tokenize(text)
        token_ids = []
        learning_rate = 0.1  # Adjust as needed
        trigrams_to_update = {}
        for token in tokens:
            if token in self.encoder:
                token_ids.append(self.encoder[token])
            else:
                trigrams = self.generate_trigrams(token)
                for trigram in trigrams:
                    trigram_id = self.get_trigram_id(trigram)
                    token_ids.append(trigram_id)

                    # Collect frequencies to update
                    trigrams_to_update[trigram] = (
                        trigrams_to_update.get(trigram, 0) + learning_rate
                    )

                    # Update reverse mapping
                    if trigram_id not in self.trigram_id_to_trigrams:
                        self.trigram_id_to_trigrams[trigram_id] = set()
                    self.trigram_id_to_trigrams[trigram_id].add(trigram)

        # Bulk update frequencies
        for trigram, freq in trigrams_to_update.items():
            self.trigram_frequencies[trigram] = (
                self.trigram_frequencies.get(trigram, 0) + freq
            )

        return token_ids

    def decay_frequencies(self):
        decay_factor = 0.9  # Adjust as needed
        # Use list to avoid RuntimeError: dictionary changed size during iteration
        trigrams = list(self.trigram_frequencies.keys())
        for trigram in trigrams:
            self.trigram_frequencies[trigram] *= decay_factor
            # Optionally remove trigrams with negligible frequencies to save memory
            if self.trigram_frequencies[trigram] < 1e-6:
                del self.trigram_frequencies[trigram]

    def reconstruct_text(self, possible_trigrams_per_token):
        epsilon = 1e-8  # Small constant to prevent division by zero
        total_freq = sum(self.trigram_frequencies.values()) + epsilon
        beam_width = 5
        sequences = [("", 0.0)]
        for trigrams in possible_trigrams_per_token:
            all_candidates = []
            for seq, score in sequences:
                for trigram in trigrams:
                    freq = self.trigram_frequencies.get(trigram, epsilon)
                    prob = freq / total_freq
                    prob = max(prob, epsilon)  # Ensure prob is positive
                    trigram_score = -math.log(prob)
                    if len(trigram) == 3:
                        new_seq = seq + trigram[-1]
                    else:
                        new_seq = seq + trigram.strip()
                    new_score = score + trigram_score
                    all_candidates.append((new_seq, new_score))
            sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]
        return sequences[0][0] if sequences else ""

    def decode(self, token_ids, **kwargs):
        # Reconstruct possible trigrams from token IDs
        possible_trigrams_per_token = []
        for token_id in token_ids:
            if token_id in self.encoder.values():
                # Handle special tokens
                token = self._convert_id_to_token(token_id)
                possible_trigrams_per_token.append({token})
            else:
                possible_trigrams = self.trigram_id_to_trigrams.get(
                    token_id, {self.unk_token}
                )
                possible_trigrams_per_token.append(possible_trigrams)

        # Reconstruct possible words from trigrams
        decoded_text = self.reconstruct_text(possible_trigrams_per_token)
        return decoded_text

    def generate_trigrams(self, word):
        padded_word = f" {word} "
        trigrams = set()
        length = len(padded_word)
        # Generate trigrams
        trigrams.update(padded_word[i : i + 3] for i in range(length - 2))
        # Generate bigrams
        trigrams.update(padded_word[i : i + 2] for i in range(length - 1))
        # Generate unigrams
        trigrams.update(padded_word[i] for i in range(length))
        return trigrams

    def get_trigram_id(self, trigram):
        hasher = self.hash_function()
        hasher.update(trigram.encode("utf-8"))
        trigram_id = int(hasher.hexdigest(), 16) % (self.config.vocab_size - 2) + 2
        return trigram_id

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        else:
            return token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory, filename_prefix=None):
        pass

    @property
    def vocab_size(self):
        return self.config.vocab_size  # Use config.vocab_size

    def get_vocab(self):
        return self.encoder.copy()

    def _tokenize(self, text):
        tokens = re.findall(r"\w+|[^\s\w]", text, re.UNICODE)
        return tokens


class TFreeModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        lm_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=config.embedding_dim,
            n_layer=3,
            n_head=4,
        )
        self.language_model = GPT2LMHeadModel(lm_config)
        self.language_model.set_input_embeddings(self.embedding)

    def forward(self, input_ids, labels=None):
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self.language_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs


def generate_random_sequence(length):
    return "".join(random.choice(string.ascii_letters + " ") for _ in range(length))


# Function to calculate similarity between two texts
def calculate_similarity(original, decoded):
    # Simple character-level similarity
    matches = sum(o == d for o, d in zip(original, decoded))
    return matches / max(len(original), len(decoded))


def truncate_text(text, max_length=512):
    return text[:max_length]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datasets import load_dataset

    config = TFreeConfig(
        vocab_size=1024,  # Increased vocab_size to reduce hash collisions
        embedding_dim=256,
        m=8,
        k=3,
    )

    tokenizer = TFreeTokenizer(config)
    model = TFreeModelForCausalLM(config)

    # Adjust as needed
    num_iterations = 10_000
    buffer_size = 1000
    # Number of test samples per iteration
    num_test_samples = 2

    # Test the tokenizer
    text = "Hello, how are you?"
    print("\nEncoding text:", text)
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)
    input_ids = torch.tensor([encoded], dtype=torch.long)
    print("Input IDs shape:", input_ids.shape)

    # Decode the tokens
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    # Forward pass
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    print("Loss:", loss.item())

    # Load the dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir="tmp",
    ).shuffle(
        buffer_size=buffer_size,
    )

    column = "text"
    dataset_iterator = (item[column] for item in dataset)

    # Initialize a list to store iteration, sample index, and similarity
    all_similarity_scores = []

    # Training and testing loop
    for iteration in range(num_iterations):
        # Get the next training sample
        train_sample = truncate_text(next(dataset_iterator))
        print(f"Iteration {iteration + 1}/{num_iterations}")

        # If the iterator is exhausted, break the loop
        if not train_sample:
            print("Training data iterator exhausted.")
            break

        # Train on train_sample
        tokenizer.encode(train_sample)

        # Apply decay after encoding the sample
        tokenizer.decay_frequencies()

        # Randomly select test samples from the testing_samples
        random_samples = []
        for i in range(num_test_samples):
            random_samples.append(next(dataset_iterator))

        # Inside your training loop
        iteration_similarity = []
        test_samples = random.sample(random_samples, num_test_samples)
        for sample_idx, test_data in enumerate(test_samples):
            encoded = tokenizer.encode(truncate_text(test_data))
            decoded = tokenizer.decode(encoded)
            similarity = calculate_similarity(test_data, decoded)
            iteration_similarity.append(similarity)
            all_similarity_scores.append(
                {
                    "Iteration": iteration + 1,  # Use capital 'I' to match the pivot
                    "SampleIndex": sample_idx,  # Include 'SampleIndex'
                    "Similarity": similarity,
                    "Sample": test_data,
                }
            )

    import pandas as pd

    # Pivot the DataFrame to create a matrix for surface plotting
    df = pd.DataFrame(all_similarity_scores)

    # Spiral plot
    theta = np.linspace(0, 4 * np.pi, len(df))
    z = df["Similarity"].values
    r = df["Iteration"].values

    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Similarity")
    plt.title("Helix")
    fig.colorbar(sc, label="Similarity")
    plt.show()

    import seaborn as sns

    # Pivot the DataFrame to create a matrix for surface plotting
    pivot_df = df.pivot(index="SampleIndex", columns="Iteration", values="Similarity")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="viridis")
    plt.xlabel("Iteration")
    plt.ylabel("Sample Index")
    plt.title("Heatmap of Similarity Scores")
    plt.show()

    # Prepare data for surface plot
    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    # Create a surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sample Index")
    ax.set_zlabel("Similarity Score")
    plt.title("Similarity Surface Over Iterations and Samples")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
