# llama.py
import time
import glob
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple
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
    rope_theta: float
    rope_traditional: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(axis=-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(input_dims=args.dim, output_dims=args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(input_dims=args.dim, output_dims=args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(input_dims=args.dim, output_dims=args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(input_dims=args.n_heads * args.head_dim, output_dims=args.dim, bias=False)
        self.rope = nn.RoPE(dims=args.head_dim, traditional=args.rope_traditional, base=args.rope_theta)


    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None
        ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # x: (batch, seq_len, dim)
        # mask: (batch, seq_len, seq_len)
        # cache: (k, v) -> (batch, seq_len, n_kv_heads, head_dim)
        B, L, D = x.shape

        # queries: wq(n_heads * head_dim, dim) * x(batch, seq_len, dim) = (batch, seq_len, n_heads * head_dim)
        # keys: wk(n_kv_heads * head_dim, dim) * x(batch, seq_len, dim) = (batch, seq_len, n_kv_heads * head_dim)
        # values: wv(n_kv_heads * head_dim, dim) * x(batch, seq_len, dim) = (batch, seq_len, n_kv_heads * head_dim)
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        # (batch, seq_len, n_heads * head_dim).reshape(B, L, self.n_heads, -1) 
        # -> (batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        # -> (batch, n_heads, seq_len, head_dim)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        # (batch, seq_len, n_kv_heads * head_dim).reshape(B, L, self.n_kv_heads, -1)
        # -> (batch, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        # -> (batch, n_kv_heads, seq_len, head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        # (batch, seq_len, n_kv_heads * head_dim).reshape(B, L, self.n_kv_heads, -1)
        # -> (batch, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        # -> (batch, n_kv_heads, seq_len, head_dim)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            # a: (batch, n_kv_heads, seq_len, head_dim)
            # mx.expand_dims(a, 2): (batch n_kv_heads, 1, seq_len, head_dim)
            # mx.concatenate(..., axis=2): (batch, n_kv_heads, repeats, seq_len, head_dim)
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape(newshape=[B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            # cache: (k, v) -> 2 x (batch, n_kv_heads, cache_len, head_dim)
            key_cache, value_cache = cache
            # (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_kv_heads, seq_len, head_dim)
            queries = self.rope(x=queries, offset=key_cache.shape[2])
            # (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_kv_heads, seq_len, head_dim)
            keys = self.rope(x=keys, offset=key_cache.shape[2])
            # (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_kv_heads, seq_len, head_dim)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(x=queries)
            keys = self.rope(x=keys)

        # (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, seq_len) -> (batch, n_heads, seq_len, seq_len)
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask

        # (batch, n_heads, seq_len, seq_len) -> (batch, n_heads, seq_len, seq_len)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)

        # scores @ values -> (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim)
        # (...).transpose(0, 2, 1, 3) -> (batch, seq_len, n_heads, head_dim)
        # (...).reshape(B, L, -1) -> (batch, seq_len, dim)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(newshape=[B, L, -1])
        # wo(dim, dim) * output(batch, seq_len, dim) -> (batch, seq_len, dim)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dims=args.hidden_dim, output_dims=args.dim, bias=False)
        self.w3 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq_len, dim)
        # nn.silu(self.w1(x)): w1(hidden_dim, dim) * x(batch, seq_len, dim) x -> (batch, seq_len, hidden_dim)
        # self.w3(x):  w3(hidden_dim, dim) * x(batch, seq_len, dim)  -> (batch, seq_len, hidden_dim)
        # nn.silu(self.w1(x)) * self.w3(x) -> (batch, seq_len, hidden_dim) * (batch, seq_len, hidden_dim)
        # -> w2(dim, hidden_dim) * (batch, seq_len, hidden_dim)  -> (batch, seq_len, dim)
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args=args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Tuple[mx.array, mx.array]] = None
        ) -> mx.array:
        # x: (batch, seq_len, dim)
        # mask: (batch, seq_len, seq_len)
        # cache: (k, v) -> (batch, seq_len, n_kv_heads, head_dim)
        r, cache = self.attention(x=self.attention_norm(x), mask=mask, cache=cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(num_embeddings=args.vocab_size, dims=args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(dims=args.dim, eps=args.norm_eps)
        self.output = nn.Linear(input_dims=args.dim, output_dims=args.vocab_size, bias=False)


    def __call__(self, x):
        # x: (batch, seq_len)
        B, L, D = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N=L)
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        # (batch, seq_len) -> (batch, seq_len, dim)
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x, _ = layer(x=x, mask=mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, x, temp = 1.0):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        caches = []

        # Make an additive causal mask, needed for processing the prompt
        mask = nn.MultiHeadAttention.create_additive_causal_mask(N=x.shape[1])
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        # First we process the prompt x the same way as in the __call__
        # save the caches in cache
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x, c = layer(x=x, mask=mask)
            # We store the cache for each layer
            caches.append(c)
        x = self.norm(x)
        # Only the last logits are needed
        y = self.output(x[:, -1])
        y = sample(y)

        # y now has the size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and continue generating
        # the rest of the sequence.
        while True:
            # Unsqueeze the last dimension to add a sequence dimension
            # of 1
            y = y[:, None]

            x = self.tok_embeddings(y)
            for layer, cache in zip(self.layers, caches):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding
                # the old cache the moment it is not needed anymore.
                x, cache = layer(x=x, mask=mask, cache=cache)
            x = self.norm(x)
            y = sample(self.output(x[:, -1]))

            yield y


def tic():
    return time.time()

def toc(msg, start):
    end = time.time()
    return (f"[INFO] {msg}: {end - start:.3f} seconds")


def generate(args, model, tokenizer):
    input("Press Enter to start generation")
    print("---------")
    print(args.prompt)
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(args.prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(x=x, temp=args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually, perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= args.max_tokens:
            break

        elif (len(tokens) % args.write_every) == 0:
            mx.eval(token)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("---------")
    print(prompt_processing)
    print(full_gen)


def sanitize_config(config, weights):
    config.pop("model_type", None)
    n_heads = config["n_heads"]
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "hidden_dim" not in config:
        # TODO: Check if this is correct
        config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
    if "vocab_size" not in config:
        config["vocab_size"] = weights["output.weight"].shape[0]
    if "rope_theta" not in config:
        config["rope_theta"] = 10_000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config


def load_model(model_path):
    model_path = Path(model_path)

    unsharded_weights_path = Path(model_path / "weights.npz")
    if unsharded_weights_path.is_file():
        print("[INFO] Loading model from {}".format(unsharded_weights_path))
        weights = mx.load(str(unsharded_weights_path))
    else:
        sharded_weights_glob = str(model_path / "weights.*.npz")
        weight_files = glob.glob(sharded_weights_glob)

        if len(weight_files) == 0:
            raise FileNotFoundError("No weights found at {}".format(model_path))

        print("[INFO] Loading model from {} shards".format(len(weight_files)))

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf).items())

    # Load the model arguments from config.json file
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config = sanitize_config(config, weights)
        quantization = config.pop("quantization", None)

    print("[INFO] Model config: {}".format(config))
    model = Llama(args=ModelArgs(**config))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model=model,
            group_size=quantization["group_size"],
            bits=quantization["bits"]
        )
    weights = tree_unflatten((list(weights.items())))
    model.update(weights)
    tokenizer = SentencePieceProcessor(model_file=str(model_path / "tokenizer.model"))
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="mlx_model",
        type=str,
        help="Path to the model weights and tokenizer"
    )
    parser.add_argument(
        "--prompt",
        default="In the beginning the universe was created",
        type=str,
        help="The prompt to be used for generation. Ignored if --few-shot is provided"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        default=100,
        type=int,
        help="The maximum number of tokens to generate"
    )
    parser.add_argument(
        "--write-every",
        default=1,
        type=int,
        help="Number of tokens to generate before detokenization and printing"
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
    generate(args, model, tokenizer)
