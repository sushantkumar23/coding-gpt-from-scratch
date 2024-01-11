# mixtral.py

import argparse
import glob
import json
from pathlib import Path
from dataclasses import dataclass

import mlx.core as mx


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
