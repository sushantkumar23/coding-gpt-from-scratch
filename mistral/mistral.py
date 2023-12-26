# mistral.py
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

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
    rope_theta: float = 10_000


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


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
        self.rope = nn.RoPE(dims=args.head_dim, base=args.rope_theta, traditional=False)

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None
        ):
        # (batch, seq_len, dims) -> (batch, seq_len, dims)
        B, L, D = x.shape

        # queries: x(batch, seq_len, dims) x wq(dims, n_heads * head_dim) -> (batch, seq_len, n_heads * head_dim)
        # keys: x(batch, seq_len, dims) x wk(dims, n_kv_heads * head_dim) -> (batch, seq_len, n_kv_heads * head_dim)
        # values: x(batch, seq_len, dims) x wv(dims, n_kv_heads * head_dim) -> (batch, seq_len, n_kv_heads * head_dim)
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1). transpose(0, 2, 1, 3)

        def repeat(a):
            # a: (batch, n_kv_heads, seq_len, head_dim)
            # mx.expand_dims(a, 2): (batch, n_kv_heads, 1, seq_len, head_dim)
            # mx.concatenate(..., axis=2): (batch, n_kv_heads, repeats, seq_len, head_dim)
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)

            # a.reshape([batch, self.n_heads, seq_len, -1]): (batch, n_heads, seq_len, head_dim)
            return a.reshape([B, self.n_heads, L, -1])

        key, values = map(repeat, (keys, values))

        if cache is not None:
            # (batch, num_heads, seq_len, head_dim) -> 2 x (batch, num_heads cache_len, head_dim)
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            # (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len + cache_len, head_dim)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len) -> (batch, num_heads, seq_len, seq_len)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        
        if mask is not None:
            scores = scores + mask

        # (batch, num_heads, seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)

        # scores @ values: (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        # (...).transpose(0, 2, 1, 3): (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        # (...).reshape(B, L, -1): (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, dims=num_heads * head_dim)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.w1 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dims=args.hidden_dim, output_dims=args.dim, bias=False)
        self.w3 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)

    def __call__(self, x: mx.array):
        # x: (batch, seq_len, dim)
        # nn.silu(self.w1(x)): x(batch, seq_len, dim) x w1(dim, hidden_dim) -> (batch, seq_len, hidden_dim)
        # self.w3(x): x(batch, seq_len, dim) x w3(dim, hidden_dim) -> (batch, seq_len, hidden_dim)
        # inner product:
        # (... * self.w3(x)): (batch, seq_len, hidden_dim) x (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
        # self.w2(...): (batch, seq_len, hidden_dim) x w2(hidden_dim, dim) -> (batch, seq_len, dim)
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))



class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.ff_norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> mx.array:
        r, cache = self.attention(x=self.attention_norm(x), mask=mask, cache=cache)
        h = x + r
        r = self.feed_forward(self.ff_norm(h))
        out = h + r
        return out, cache


class Mistral(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(num_embeddings=self.vocab_size, dims=args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(self.n_layers)]
        self.norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.output = nn.Linear(input_dims=args.dim ,output_dims=args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache = None
    ):
        # inputs: (batch, seq_len)
        # h: (batch, seq_len, dim)
        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * (len(self.layers))

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), cache


class Tokenizer(nn.Module):
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "_"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out


def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    model = Mistral(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model=model,
            group_size=quantization["group_size"],
            bits=quantization["bits"]
        )
    model.update(weights)
    mx.eval(model.parameters())
    return model, tokenizer


def generate(prompt: mx.array, model: Mistral, temp: Optional[float] = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))
    
    logits, cache = model(inputs=prompt[None])
    y = sample(logits[:, -1, :])
    yield y

    while True:
        logits, cache = model(inputs=y[:, None], cache=cache)
        y = sample(logits.squeeze(1))
        yield y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral inference script")
    parser.add_argument(
        "--model-path",
        default="mlx_model",
        type=str,
        help="The path to the model weights and tokenizer"
    )
    parser.add_argument(
        "--prompt",
        default="In the beginning, the universe was created.",
        type=str,
        help="The prompt to be processed by the model"
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
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--tokens_per_eval",
        default=10,
        type=int,
        help="The batch size of tokens to generate"
    )
    parser.add_argument("--seed", default=0, type=int, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)
    print("[INFO] Loading the model from the disk")
    model, tokenizer = load_model(args.model_path)

    print("[INFO] Starting the generation process")
    tic = time.time()
    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))
    tokens = []
    for token, ntoks in zip(generate(prompt, model, args.temp), range(args.max_tokens)):
        tokens.append(token)

        if ntoks == 0:
            mx.eval(tokens)
            toc = time.time()
            prompt_tps = prompt.size / (toc - tic)
            tic = time.time()

        if (len(tokens) % args.tokens_per_eval) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
    
    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)
    print("-------")
    generation_tps = ntoks / (time.time() - toc)
    print(f"Prompt TPS: {prompt_tps:.3f}")
    print(f"Generation TPS: {generation_tps:.3f}")
