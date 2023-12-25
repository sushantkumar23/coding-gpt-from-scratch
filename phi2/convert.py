# conver.py
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_map, tree_unflatten, tree_flatten
from transformers import AutoModelForCausalLM

from phi2 import ModelArgs, Phi2

def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model
    model = Phi2(ModelArgs())
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model
    nn.QuantizedLinear.quantize_module(model, args.q_group_size, args.q_bits)

    # Update the config
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def replace_key(key: str) -> str:
    if "wte.weight" in key:
        key = "wte.weight"

    if ".mlp" in key:
        key = key.replace(".mlp", "")
    return key


def convert():
    parser = argparse.ArgumentParser(description="Convert Phi-2 weights to MLX")
    parser.add_argument(
        "--mlx-path",
        default="mlx_model",
        type=str,
        help="Path to save the MLX model"
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model",
        action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        default=64,
        type=int,
        help="Group size for quantization"
    )
    parser.add_argument(
        "--q-bits",
        default=4,
        type=int,
        help="Bits per weight for quantization"
    )
    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    state_dict = model.state_dict()
    weights = { replace_key(k): v.numpy() for k, v in state_dict.items() }
    params = {}
    if args.quantize:
        print("[INFO] Quantizing model...")
        weights, params = quantize(weights, params, args)

    np.savez(str(mlx_path / "weights.npz"), **weights)
    with open(mlx_path / "config.json", "w") as fid:
        params["model_type"] = "phi2"
        json.dump(params, fid, indent=4)


if __name__ == "__main__":
    convert()
