import mmh3  # MurmurHash3, a fast non-cryptographic hash function


class RollingHash:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []
        self.hash = 0

    def append(self, token):
        if len(self.window) == self.window_size:
            self.hash ^= mmh3.hash(self.window[0])
            self.window.pop(0)
        self.window.append(token)
        self.hash ^= mmh3.hash(token)

    def get_hash(self):
        return self.hash


def detect_advanced_repetition(
    text, window_size=5, threshold=3, similarity_threshold=0.7, max_gap=100
):
    """
    Detect repetition using an advanced rolling hash method with fuzzy matching.

    :param text: The input string or list of tokens to check for repetition
    :param window_size: Size of the rolling window (default is 5)
    :param threshold: Number of similar windows to consider as significant repetition (default is 3)
    :param similarity_threshold: Jaccard similarity threshold for considering windows as similar (default is 0.7)
    :param max_gap: Maximum gap between repeats to consider as significant (default is 100)
    :return: True if repetition is detected, False otherwise
    """
    tokens = text.split() if isinstance(text, str) else text
    rh = RollingHash(window_size)
    seen = {}

    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        for token in window:
            rh.append(token)
        h = rh.get_hash()

        if h in seen:
            for prev_i in seen[h]:
                if i - prev_i <= max_gap:
                    prev_window = tokens[prev_i : prev_i + window_size]
                    similarity = jaccard_similarity(set(window), set(prev_window))
                    if similarity >= similarity_threshold:
                        seen[h].append(i)
                        if len(seen[h]) >= threshold:
                            print(
                                f"Repetition detected! Similar windows found at positions {seen[h]}"
                            )
                            return True
        else:
            seen[h] = [i]

    print("No significant repetition detected.")
    return False


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


# Example usage
text1 = "The cat sat on the mat. The dog sat on the rug. The cat lay on the mat."
print(
    "Advanced rolling hash result for text1:",
    detect_advanced_repetition(text1, 20, 5, 0.7, 100),
)

text2 = "The quick brown fox jumps over the lazy dog. A fast orange fox leaps above a sleepy hound."
print("\nAdvanced rolling hash result for text2:", detect_advanced_repetition(text2))

# class RollingHash:
#     def __init__(self, window_size):
#         self.window_size = window_size
#         self.hash = 0
#         self.text = ""
#         self.BASE = 256
#         self.MOD = 2**32

#     def append(self, c):
#         self.hash = (self.hash * self.BASE + ord(c)) % self.MOD
#         self.text += c
#         if len(self.text) > self.window_size:
#             self.hash = (
#                 self.hash
#                 - ord(self.text[0]) * pow(self.BASE, self.window_size - 1, self.MOD)
#             ) % self.MOD
#             self.text = self.text[1:]

#     def get_hash(self):
#         return self.hash


# def detect_repetition_ngram(text, n_gram_size=10, threshold=2):
#     """
#     Detect repetition in a string using n-gram analysis.

#     :param text: The input string to check for repetition
#     :param n_gram_size: Size of n-grams to use (default is 10)
#     :param threshold: Number of repeats to consider as significant repetition (default is 2)
#     :return: True if repetition is detected, False otherwise
#     """
#     n_grams = [text[i : i + n_gram_size] for i in range(len(text) - n_gram_size + 1)]
#     n_gram_counts = {}

#     for i, n_gram in enumerate(n_grams):
#         if n_gram in n_gram_counts:
#             n_gram_counts[n_gram].append(i)
#         else:
#             n_gram_counts[n_gram] = [i]

#     for n_gram, positions in n_gram_counts.items():
#         if len(positions) >= threshold:
#             print(
#                 f"Repetition detected! N-gram '{n_gram}' repeated {len(positions)} times at positions {positions}"
#             )
#             return True

#     print("No significant repetition detected.")
#     return False


# def detect_repetition_rolling(text, window_size=20, threshold=2, max_gap=50):
#     """
#     Detect repetition using rolling hash method.

#     :param text: The input string to check for repetition
#     :param window_size: Size of the rolling window (default is 20)
#     :param threshold: Number of repeats to consider as significant repetition (default is 2)
#     :param max_gap: Maximum gap between repeats to consider as significant (default is 50)
#     :return: True if repetition is detected, False otherwise
#     """
#     rh = RollingHash(window_size)
#     seen = {}

#     for i in range(len(text) - window_size + 1):
#         window = text[i : i + window_size]
#         for c in window:
#             rh.append(c)
#         h = rh.get_hash()

#         if h in seen:
#             seen[h].append(i)
#             if len(seen[h]) >= threshold:
#                 repeats = seen[h]
#                 if any(
#                     repeats[j] - repeats[j - 1] <= max_gap
#                     for j in range(1, len(repeats))
#                 ):
#                     print(
#                         f"Repetition detected! Window '{window}' repeated {len(repeats)} times at positions {repeats}"
#                     )
#                     return True
#         else:
#             seen[h] = [i]

#     print("No significant repetition detected.")
#     return False


# # Example usage
# text1 = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
# print("N-gram result for text1:", detect_repetition_ngram(text1))
# print("\nRolling hash result for text1:", detect_repetition_rolling(text1))

# print("\n" + "=" * 50 + "\n")

# text2 = "The quick brown fox jumps over the lazy dog."
# print("N-gram result for text2:", detect_repetition_ngram(text2))
# print("\nRolling hash result for text2:", detect_repetition_rolling(text2))

# import zlib


# def detect_repetition_compression(text, threshold=0.5):
#     """
#     Detect repetition using compression ratio.

#     :param text: The input string to check for repetition
#     :param threshold: Compression ratio threshold (default is 0.5)
#     :return: True if repetition is detected, False otherwise
#     """
#     compressed = zlib.compress(text.encode("utf-8"))
#     compression_ratio = len(compressed) / len(text)

#     return compression_ratio < threshold


# # Example usage
# text = "The cat sat on the mat. " * 10
# print(detect_repetition_compression(text))  # True

# text = "The quick brown fox jumps over the lazy dog."
# print(detect_repetition_compression(text))  # False
