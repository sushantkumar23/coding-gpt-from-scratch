# phixtral.py

import glob
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, fields

import mlx.nn as nn
import mlx.core as mx
from huggingface_hub import snapshot_download


@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32
    num_experts_per_tok: int = 2
    num_local_experts: int = 4


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class Embd(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(num_embeddings=config.num_vocab, embedding_dim=config.model_dim)


class RoPEAttention(nn.Module):

    def __init__(self, dims: int, num_heads: int, rotary_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.rope = nn.RoPE(dims=rotary_dim, traditional=False)
        self.Wqkv = nn.Linear(input_dims=dims, output_dims=(dims * 3))
        self.out_proj = nn.Linear(input_dims=dims, output_dims=dims)


class MLP(nn.Module):
    def __init__(self, dims, hidden_dims):
        super().__init__()
        self.fc1 = nn.Linear(input_dims=dims, output_dims=hidden_dims)
        self.fc2 = nn.Linear(input_dims=hidden_dims, output_dims=dims)
        self.act = nn.GELU(approx="precise")


class MOE(nn.Module):

    def __init__(self, config: ModelArgs, dims: int, hidden_dims: int):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.mlp = [MLP(dims=dims, hidden_dims=hidden_dims) for _ in range(self.num_experts)]
        self.gate = nn.Linear(input_dims=config.model_dim, output_dims=self.num_experts, bias=False)


class ParallelBlock(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.mixer = RoPEAttention(
            dims=config.model_dim,
            num_heads=config.num_heads,
            rotary_dim=config.rotary_dim
        )
        self.ln = LayerNorm(dims=config.model_dim)
        self.moe = MOE(args=config, dims=config.model_dim, hidden_dims=(config.model_dim * 4))


class TransformerDecoder(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embd = Embd(config)
        self.h = [ParallelBlock(config) for _ in range(config.num_layers)]


class OutputHead(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.ln = LayerNorm(dims=args.model_dim)
        self.linear = nn.Linear(input_dims=args.model_dim, output_dims=args.num_vocab, bias=False)


class Phixtral(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.transformer = TransformerDecoder(config)
        self.lm_head = OutputHead(config)


def load_model(path_or_repo_id: str):
    """
    Args:
        model_path: The path to the model directory or huggingface repo.
    """
    model_path = Path(path_or_repo_id)

    # If the model path doesn't exist, try to download it from Hugging Face.
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_repo_id,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
                local_dir=model_path
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

        model_args_fields = { field.name for field in fields(ModelArgs) }
        filtered_config = { k: v for k, v in config.items() if k in model_args_fields }
        model_args = ModelArgs(**filtered_config)
        print("Model Args:", model_args)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise ValueError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items)

    model = Phixtral(model_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phixtral Inference Script")
    parser.add_argument(
        "--model-path",
        default="mlx_model",
        type=str,
        help="The path to the local model directory"
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The PRNG seed"
    )

    args = parser.parse_args()
    mx.random.seed(args.seed)
    load_model(path_or_repo_id=args.model_path)
