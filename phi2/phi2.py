# phi2.py
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32


class LayerNorm(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class RoPEAttention(nn.Module):

    def __init__(self, dims: int, num_heads: int, rotary_dim: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims=rotary_dim, traditional=False)
        self.Wqkv = nn.Linear(input_dims=dims, output_dims= 3 * dims)
        self.out_proj = nn.Linear(input_dims=dims, output_dims=dims)

    def __call__(self, x, mask=None, cache=None):
        # x: [batch, seq_len, dims]
        # (batch, seq_len, dims) -> (batch, seq_len, 3 * dims))
        qkv = self.Wqkv(x)
        # (batch, seq_len, 3 * dims) -> 3 * (batch, seq_len, dims)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
        # (batch, seq_len, dims) -> (batch, seq_len, dims)
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computations
        # (batch, seq_len, dims) -> (batch, seq_len, num_heads, dims // num_heads)
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to queries and keys and combine them with cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            
            # (batch, num_heads, seq_len, dims // num_heads) -> (batch, num_heads, seq_len + cache_len, dims // num_heads)
            keys = mx.concatenate([key_cache, keys], axis=2)
            # (batch, num_heads, seq_len, dims // num_heads) -> (batch, num_heads, seq_len + cache_len, dims // num_heads)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            # (batch, num_heads, seq_len, dims // num_heads) -> (batch, num_heads, seq_len, dims // num_heads)
            queries = self.rope(queries)
            # (batch, num_heads, seq_len, dims // num_heads) -> (batch, num_heads, seq_len, dims // num_heads)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])

        # (batch, num_heads, seq_len, dims // num_heads) * (batch, num_heads, dims // num_heads, seq_len) -> (batch, num_heads, seq_len, seq_len)
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, -1).astype(values.dtype)

        # (batch, num_heads, seq_len, seq_len) * (batch, num_heads, seq_len, dims // num_heads) -> (batch, num_heads, seq_len, dims // num_heads).transpose(0, 2, 1, 3) -> (batch, seq_len, num_heads, dims // num_heads).reshape(B, L, -1) -> (batch, seq_len, dims)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # (batch, seq_len, dims) -> (batch, seq_len, dims)
        return self.out_proj(values_hat), (keys, values)


class ParallelBlock(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        dims = config.model_dim
        mlp_dims = dims * 4
        self.mixer = RoPEAttention(dims, config.num_heads, config.rotary_dim)
        self.ln = LayerNorm(dims)
        self.fc1 = nn.Linear(input_dims=dims, output_dims=mlp_dims)
        self.act = nn.GELU(approx="precise")
        self.fc2 = nn.Linear(input_dims=mlp_dims, output_dims=dims)

    def __call__(self, x, mask, cache):
        h = self.ln(x)
        attn_h, cache = self.mixer(h, mask, cache)
        ff_h = self.fc2(self.act(self.fc1(h)))
        return attn_h + ff_h + x, cache


class TransformerDecoder(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.h = [ParallelBlock(config) for i in range(config.num_layers)]

    def __call__(self, x, mask, cache):
        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])

        return x, cache


class OutputHead(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        self.ln = LayerNorm(config.model_dim)
        self.linear = nn.Linear(input_dims=config.model_dim, output_dims=config.num_vocab)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.ln(x))


class Phi2(nn.Module):

    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(num_embeddings=config.num_vocab, dims=config.model_dim)
        self.transformer = TransformerDecoder(config=config)
        self.lm_head = OutputHead(config=config)

    def __call__(
            self,
            inputs: mx.array,
            mask: mx.array = None,
            cache: mx.array = None
        ) -> tuple[mx.array, mx.array]:

        # (batch, seq_len) -> (batch, seq_len, dims)
        x = self.wte(inputs)

        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(N=x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        return self.lm_head(y), cache


def generate(prompt: mx.array, model: Phi2, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0.0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    # (batch, seq_len) -> (batch, seq_len, num_vocab)
    logits, cache = model(prompt)
    # (batch, seq_len, num_vocab) -> (batch)
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(inputs=y[:, None], cache=cache)
        y = sample(logits.squeeze(1))
        yield y


def load_model(model_path: str):
    model = Phi2(ModelArgs())
    model_path = Path(model_path)
    with open(model_path / "config.json", "r") as f:
        config = json.load(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(weights)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phi-2 inference script")
    parser.add_argument(
        "--model-path",
        default="mlx_model",
        type=str,
        help="Path to model weights"
    )
    parser.add_argument(
        "--prompt",
        default="Write a detailed analogy between mathematics and a lighthouse",
        help="Message to be processed by the model"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        default=100,
        type=int,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temp",
        default=0.0,
        type=float,
        help="The sampling temperature"
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The PRNG seed"
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path)

    prompt = tokenizer(
        text=args.prompt,
        return_tensors="np",
        return_attention_mask=False
    )["input_ids"]

    prompt = mx.array(prompt)

    print("[INFO] Generating with Phi-2...", flush=True)
    print(args.prompt, end="", flush=True)

    tokens = []
    for token, _ in zip(generate(prompt, model, args.temp), range(args.max_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            eos_index = next(
                (i for i, t in enumerate(tokens) if t.item() == tokenizer.eos_token_id),
                None,
            )

            if eos_index is not None:
                tokens = tokens[:eos_index]

            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
            if eos_index is not None:
                break

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
