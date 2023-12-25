# model.py
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the keys and values
    vocab_size: int = -1  # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1


class EncoderBlock(nn.Module):

    def __init__(self):
        super().__init__()


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paper, the dimension of the embedding must be even
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula: theta_i = 10000 ^ (-2(i-1)/d) for i 
    # shape: (head_dim // 2)
    theta_pos = torch.arange(start=0, end=head_dim, step=2, device=device).float()



class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args=args))

        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.output = nn.Linear(in_features=args.dim, out_features=self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_len) -> (batch, seq_len)
        batch_size, seq_len, _ = tokens.shape()

        assert seq_len == 1, "Only one token at a time can be processed"

        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # (batch, seq_len, dim) -> (batch, seq_len, dim)
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output
