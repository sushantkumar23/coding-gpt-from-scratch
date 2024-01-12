# phixtral.py

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, fields

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
    num_experts_per_token: int = 2
    num_local_experts: int = 4


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
