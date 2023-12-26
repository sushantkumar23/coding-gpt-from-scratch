# llama.py
from typing import Optional, Tuple
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

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

        # queries: x(batch, seq_len, dim) x wq(dim, n_heads * head_dim) = (batch, seq_len, n_heads * head_dim)
        # keys: x(batch, seq_len, dim) x wk(dim, n_kv_heads * head_dim) = (batch, seq_len, n_kv_heads * head_dim)
        # values: x(batch, seq_len, dim) x wv(dim, n_kv_heads * head_dim) = (batch, seq_len, n_kv_heads * head_dim)
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
        # (batch, seq_len, dim) x wo(dim, dim) -> (batch, seq_len, dim)
        return self.wo(output), (keys, values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)
        self.w2 = nn.Linear(input_dims=args.hidden_dim, output_dims=args.dim, bias=False)
        self.w3 = nn.Linear(input_dims=args.dim, output_dims=args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq_len, dim)
        # nn.silu(self.w1(x)): (batch, seq_len, dim) x w1(dim, hidden_dim) -> (batch, seq_len, hidden_dim)
        # self.w3(x): (batch, seq_len, dim) x w3(dim, hidden_dim) -> (batch, seq_len, hidden_dim)
        # nn.silu(self.w1(x)) * self.w3(x) -> (batch, seq_len, hidden_dim) * (batch, seq_len, hidden_dim)
        # -> (batch, seq_len, hidden_dim) x w2(hidden_dim, dim) -> (batch, seq_len, dim)
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

        cache = []

