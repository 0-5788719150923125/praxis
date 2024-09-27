import hashlib
import math
import random
import re
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedTokenizer


class PraxisTokenizerConfig(PretrainedConfig):
    model_type = "t_free"

    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=768,
        num_activations=8,
        lowercase_overlap=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_activations = num_activations
        self.lowercase_overlap = lowercase_overlap
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3


class PraxisTokenizer(PreTrainedTokenizer):
    def __init__(self, config):
        self.config = config
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.encoder = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
        )

        self.pad_token_id = self.encoder[self.pad_token]
        self.unk_token_id = self.encoder[self.unk_token]
        self.bos_token_id = self.encoder[self.bos_token]
        self.eos_token_id = self.encoder[self.eos_token]

        self.hash_function = hashlib.sha256
        self.num_activations = config.num_activations
        self.lowercase_overlap = config.lowercase_overlap

        self.trigram_id_to_trigram = {}
        self.trigram_weights = {}
        self.trigram_positions = {}  # New attribute to store trigram positions

    def encode(self, text, add_special_tokens=True, **kwargs):
        tokens = self._tokenize(text)
        token_ids = []
        position = 0

        if add_special_tokens:
            token_ids.append(self.bos_token_id)
            position += 1

        for token in tokens:
            if token in self.encoder:
                token_ids.append(self.encoder[token])
                position += 1
            else:
                trigrams = self.generate_trigrams(token)
                for trigram in trigrams:
                    trigram_id, weight = self.get_trigram_id_and_weight(trigram, token)
                    token_ids.append(trigram_id)
                    self.trigram_id_to_trigram[trigram_id] = trigram
                    self.trigram_weights[trigram] = weight
                    self.trigram_positions[trigram_id] = position
                    position += 1

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if skip_special_tokens:
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id
                not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ]

        decoded_tokens = []
        current_word_trigrams = []
        current_word_positions = []

        for token_id in token_ids:
            if token_id in self.encoder.values():
                if current_word_trigrams:
                    decoded_tokens.append(
                        self.reconstruct_word(
                            current_word_trigrams, current_word_positions
                        )
                    )
                    current_word_trigrams = []
                    current_word_positions = []
                decoded_tokens.append(self._convert_id_to_token(token_id))
            else:
                trigram = self.trigram_id_to_trigram.get(token_id, "")
                position = self.trigram_positions.get(token_id, 0)
                if trigram.startswith(" ") and current_word_trigrams:
                    decoded_tokens.append(
                        self.reconstruct_word(
                            current_word_trigrams, current_word_positions
                        )
                    )
                    current_word_trigrams = []
                    current_word_positions = []
                current_word_trigrams.append(trigram)
                current_word_positions.append(position)

        if current_word_trigrams:
            decoded_tokens.append(
                self.reconstruct_word(current_word_trigrams, current_word_positions)
            )

        decoded_text = self.post_process_text(" ".join(decoded_tokens))
        return decoded_text

    def reconstruct_word(self, trigrams, positions):
        word = ""
        sorted_trigrams = sorted(zip(trigrams, positions), key=lambda x: x[1])

        for trigram, _ in sorted_trigrams:
            if not word:
                word = trigram.strip()
            else:
                overlap = self.find_overlap(word, trigram)
                word += trigram[overlap:].strip()

        return word.strip()

    def find_overlap(self, word, trigram):
        for i in range(min(len(word), 3), 0, -1):
            if word.endswith(trigram[:i].strip()):
                return i
        return 0

    def post_process_text(self, text):
        # Handle punctuation and spacing
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"([.,!?])([^\s])", r"\1 \2", text
        )  # Add space after punctuation if missing
        return text.strip()

    def generate_trigrams(self, word):
        padded_word = f" {word} "
        length = len(padded_word)
        trigrams = [padded_word[i : i + 3] for i in range(length - 2)]
        return trigrams

    def get_trigram_id_and_weight(self, trigram, word):
        position = word.find(trigram.strip())
        weight = 1.0

        if position == 0 or position == len(word) - 3:
            weight = (
                1.5  # Higher weight for trigrams at the beginning or end of the word
            )

        hash_input = (
            f"{trigram.lower()}"
            if random.random() < self.lowercase_overlap
            else f"{trigram}"
        )
        trigram_id = self.hash_trigram(hash_input)

        return trigram_id, weight

    def get_trigram_id(self, trigram):
        hash_input = (
            f"{trigram.lower()}"
            if random.random() < self.lowercase_overlap
            else f"{trigram}"
        )
        trigram_id = self.hash_trigram(hash_input)
        return trigram_id

    def get_trigram_ids(self, trigram):
        trigram_ids = []
        for i in range(self.num_activations):
            if i < self.lowercase_overlap:
                hash_input = f"{trigram.lower()}_{i}"
            else:
                hash_input = f"{trigram}_{i}"
            trigram_id = self.hash_trigram(hash_input)
            trigram_ids.append(trigram_id)
        return trigram_ids

    def hash_trigram(self, trigram):
        hasher = self.hash_function()
        hasher.update(trigram.encode("utf-8"))
        trigram_id = int(hasher.hexdigest(), 16) % (self.config.vocab_size - 4) + 4
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
        return self.config.vocab_size

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
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, labels=None):
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self.language_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits

        if labels is not None:
            # Assuming labels are multi-hot encoded
            loss = self.loss_fn(logits, labels.float())
            outputs.loss = loss

        return outputs


def generate_dictionary(tokenizer, dataset, max_words=100000):
    dictionary = defaultdict(set)
    for text in dataset:
        tokens = tokenizer._tokenize(text)
        for token in tokens:
            if token not in tokenizer.encoder:
                trigrams = tokenizer.generate_trigrams(token)
                for trigram in trigrams:
                    trigram_ids = tokenizer.get_trigram_ids(trigram)
                    dictionary[token].update(trigram_ids)
        if len(dictionary) >= max_words:
            break
    return dictionary


def calculate_similarity(original, decoded):
    matches = sum(o == d for o, d in zip(original, decoded))
    return matches / max(len(original), len(decoded))


def truncate_text(text, max_length=512):
    return text[:max_length]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from datasets import load_dataset
    from transformers import GPT2Config, GPT2LMHeadModel

    config = PraxisTokenizerConfig(
        vocab_size=1024,
        embedding_dim=256,
        num_activations=8,
        lowercase_overlap=3,
    )

    tokenizer = PraxisTokenizer(config)
    model = TFreeModelForCausalLM(config)

    num_iterations = 100
    buffer_size = 1000
    num_test_samples = 2

    text = "Hello, how are you?"
    print("\nEncoding text:", text)
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    def test_tokenizer(tokenizer, text):
        print("\nOriginal text:")
        print(text)
        print("\nEncoding text...")
        encoded = tokenizer.encode(text)
        print("Encoded:", encoded)
        print("\nDecoding...")
        decoded = tokenizer.decode(encoded)
        print("Decoded:", decoded)
        print("\nSimilarity:", calculate_similarity(text, decoded))
        print("-" * 80)

    # Initialize the tokenizer
    config = PraxisTokenizerConfig(
        vocab_size=1024,
        embedding_dim=256,
        num_activations=8,
        lowercase_overlap=3,
    )
    tokenizer = PraxisTokenizer(config)

    # Test cases
    test_cases = [
        "Hey! What's up? This sentence has multiple punctuation marks!!! Isn't it cool?",
        "Bonjour, comment allez-vous? J'espère que vous passez une bonne journée.",
        "The quick brown fox jumps over the lazy dog. This is a pangram, which means it contains every letter of the alphabet. It's often used to showcase fonts or test character encoding. The sentence's length makes it useful for testing various text-processing tasks, including tokenization and language models.",
        "In 2023, AI made significant progress. GPT-4 achieved 90% accuracy on complex tasks, while DALL-E3 generated photo-realistic images with 95% fidelity.",
        "The HTTP protocol uses TCP/IP for data transfer. APIs often return JSON or XML. CRUD operations (CREATE, READ, UPDATE, DELETE) are fundamental in RESTful services.",
    ]

    # Run tests
    for case in test_cases:
        test_tokenizer(tokenizer, case)

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

    all_similarity_scores = []

    for iteration in range(num_iterations):
        train_sample = truncate_text(next(dataset_iterator))
        print(f"Iteration {iteration + 1}/{num_iterations}")

        tokenizer.encode(train_sample)

        random_samples = [next(dataset_iterator) for _ in range(num_test_samples)]

        iteration_similarity = []
        test_samples = random.sample(random_samples, num_test_samples)
        for sample_idx, test_data in enumerate(test_samples):
            encoded = tokenizer.encode(truncate_text(test_data))
            decoded = tokenizer.decode(encoded)
            similarity = calculate_similarity(test_data, decoded)
            iteration_similarity.append(similarity)
            all_similarity_scores.append(
                {
                    "Iteration": iteration + 1,
                    "SampleIndex": sample_idx,
                    "Similarity": similarity,
                    "Sample": test_data,
                }
            )

    df = pd.DataFrame(all_similarity_scores)

    theta = np.linspace(0, 4 * np.pi, len(df))
    z = df["Similarity"].values
    r = df["Iteration"].values

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

    pivot_df = df.pivot(index="SampleIndex", columns="Iteration", values="Similarity")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, cmap="viridis")
    plt.xlabel("Iteration")
    plt.ylabel("Sample Index")
    plt.title("Heatmap of Similarity Scores")
    plt.show()

    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sample Index")
    ax.set_zlabel("Similarity Score")
    plt.title("Similarity Surface Over Iterations and Samples")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    dictionary = generate_dictionary(tokenizer, dataset[column])

    test_text = "This is a test sentence."
    encoded_ids = tokenizer.encode(test_text)
    trigram_ids = torch.tensor([encoded_ids], dtype=torch.long)
    dictionary_matrix = torch.zeros((len(dictionary), config.vocab_size))
    for idx, (word, trigrams) in enumerate(dictionary.items()):
        dictionary_matrix[idx, list(trigrams)] = 1

    with torch.no_grad():
        outputs = model(trigram_ids)
        logits = outputs.logits
        scores = torch.matmul(dictionary_matrix, logits.squeeze())
        predicted_idx = scores.argmax().item()
        predicted_word = list(dictionary.keys())[predicted_idx]

    print("Test Text:", test_text)
    print("Predicted Next Word:", predicted_word)
