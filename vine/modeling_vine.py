import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Create a causal mask
        seq_length = x.size(1)
        causal_mask = (torch.triu(torch.ones(seq_length, seq_length)) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        causal_mask = causal_mask.to(x.device)
        
        x = self.transformer(x, mask=causal_mask)
        x = self.output_projection(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Example usage
vocab_size = 10000
d_model = 256
nhead = 8
num_layers = 6
max_seq_length = 512

model = SimpleTransformer(vocab_size, d_model, nhead, num_layers, max_seq_length)

# Example input (batch_size=2, seq_length=10)
x = torch.randint(0, vocab_size, (2, 10))
output = model(x)
print(output.shape)  # Should be (2, 10, vocab_size)