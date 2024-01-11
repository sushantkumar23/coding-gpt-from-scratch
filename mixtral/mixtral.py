# mixtral.py

import argparse
import glob
import json
from pathlib import Path
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    moe: dict = None


class RoPE(nn.RoPE):

    def __init__(self, dims: int, traditional: bool = False):
        super().__init__(dims, traditional=traditional)


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(input_dims=args.dim, output_dims=args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(input_dims=args.dim, output_dims=args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(input_dims=args.dim, output_dims=args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(input_dims=args.n_heads * args.head_dim, output_dims=args.dim, bias=False)
        self.rope = RoPE(args.head_dim, traditional=False)



class MOETransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args=args)



class Mixtral(nn.Module):

    def __init__(self):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = [
            MOETransformerBlock(args=args) for _ in range(self.n_layers)
        ]


class Tokenizer:
    pass


def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weight_files = glob.glob(str(model_path / "weights.*.npz"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())
    weights = tree_unflatten(list(weights.items()))
    model = Mixtral(model_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixtral inference script")

    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the models weights, tokenizer and config"
    )

    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    print("[INFO] Loading model from disk...")
    model, tokenizer = load_model(args.model_path)
