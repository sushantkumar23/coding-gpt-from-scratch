# mixtral.py

import argparse
import glob
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor


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


class RMSNorm(nn.Module):

    def __init__(self, dims: float, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class RoPE(nn.RoPE):

    def __init__(self, dims: int, traditional: bool = False):
        super().__init__(dims, traditional=traditional)

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=1_000_000, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)


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

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.wq(x),
        keys = self.wk(x)
        values = self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        key, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dims=args.hidden_dim, output_dims=args.dim, bias=False)
        self.w3 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.num_experts = args.moe["num_experts"]
        self.num_experts_per_tok = args.moe["num_experts_per_tok"]
        self.experts = [FeedForward(args=args) for _ in range(self.num_experts)]
        self.gate = nn.Linear(input_dims=args.dim, output_dims=self.num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)
        inds = mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne]
        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1
        ).astype(gates.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt[None, :])
        y = mx.concatenate(y)

        return y.reshape(orig_shape)


class MOETransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args=args)
        self.feed_forward = MOEFeedForward(args=args)
        self.attention_norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


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
        self.norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.output = nn.Linear(input_dims=args.dim, output_dims=self.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.tok_embeddings(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])


class Tokenizer:

    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "_"
        assert self._model.vocab_size == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]


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
    if quantization is not None:
        quantization["linear_class_predicate"] = (
            lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != 8
        )
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.update(weights)
    return model, tokenizer


def generate(prompt: mx.array, model: Mixtral, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    logits, cache = model(prompt[None])
    y = sample(logits[:, -1, :])
    yield y


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

    print("[INFO] Starting generation...")

    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))
    tokens = []
