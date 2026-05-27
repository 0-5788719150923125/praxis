import torch.nn as nn


class ByteEmbedding(nn.Module):
    """Per-byte token table for byte-latent encoders; the weight-tying target.

    Sizes itself from the encoder's declared input layout (``input_dim`` /
    ``input_vocab_size``) when present, else from the global config.
    """

    def __init__(self, config, encoder=None):
        super().__init__()
        dim = encoder.input_dim if encoder is not None else config.embed_size
        num = encoder.input_vocab_size if encoder is not None else config.vocab_size
        self.tokens = nn.Embedding(num, dim)

    @property
    def weight(self):
        return self.tokens.weight

    def forward(self, tokens):
        return self.tokens(tokens)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab={self.tokens.num_embeddings}, dim={self.tokens.embedding_dim})"
        )
