# model.py
import os
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor


class Sampler(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ):


@dataclass
class GemmaConfig:
    # The default config is for Gemma7B

    # The size of the vocabulary
    vocab_size: int = 256_000

    # Context length: Max number of tokens in the sequence
    max_position_embeddings: int = 8192

    # Layers in the model
    num_hidden_layers: int = 28

    # Number of attention heads in the transformer
    num_attention_heads: int = 16

    # Number of KV heads in Attention
    num_key_value_heads: int = 16

    # Embedding size that is fed to the transformer
    hidden_size: int = 3072

    # Transformer: Feedforward FFN hidden dimension
    intermediate_size: int = 24576

    # Head dimensions
    head_dim: int = 256

    # Whether a quantized version of the model is used
    quant: bool = False

    # The path to model tokenizer
    tokenizer: Optional[str] = "tokenizer/tokenizer.model"


def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384,
    )


class Tokenizer:

    def __init__(self, model_path: Optional[str]):
        # Reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token ids
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """
        Converts a string to a list of token ids
        """
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Converts a list of token ids to a string
        """
        assert isinstance(t, list)
        return self.sp_model.decode(t)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors"""
    x_ = torch.view_as_complex(
        torch.stack(
            tensors=torch.chunk(input=x.transpose(1, 2).float(), chunks=2, dim=-1),
            dim=-1,
        )
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(
        1, 2
    )
    return x_out


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)), requires_grad=False
            )
        self.quant = quant


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)), requires_grad=False
            )
        self.quant = quant


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output


class GemmaMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, quant: bool):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        # Gemma7B: 3072 -> 24576 -> 3072
        # Gemma2B: 2048 -> 16384 -> 2048
        gate = self.gate_proj(x)
        gate = F.gelu(input=gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        quant: bool,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        # Gemma7B: 16 heads x 256 head_dim = 4096
        # Gemma2B: 8 heads x 256 head_dim = 2048
        self.q_size = self.num_heads * self.head_dim

        # Gemma7B: 16 KV heads x 256 head_dim = 4096
        # Gemma2B: 1 KV head x 256 head_dim = 256
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            in_features=hidden_size,
            out_features=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=quant,
        )

        self.o_proj = Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=hidden_size,
            quant=quant,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Implement: Rotary Positional Embeddings
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Implement: KV Cache
        # Write the new KV to the cache
        # [batch_size, num_kv_heads, seq_len, head_dim]
        k_cache, v_cache = kv_cache
        k_cache = k_cache.index_copy_(dim=1, index=kv_write_indices, source=xk)
        v_cache = v_cache.index_copy_(dim=1, index=kv_write_indices, source=xv)

        # Implement: MQA - Multi-Query Attention
        # Copy and repeat the key and value tensors for the queries
        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, seq_len, num_kv_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeate_interleave(key, self.num_queries_per_kv, dim=2)

        # Switch heads and sequence length
        q = xq.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # Implement: Attention
        # Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)

        # Switch back heads and sequence length
        # and concatenate the heads
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config

        # Attention Layer
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            quant=config.quant,
        )

        # Feedforward Layer
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )

        # Layer Norms
        # The input layernorm is applied for the self-attention
        self.input_layernorm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)

        # The post-attention layernorm is applied for the feedforward layer
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = hidden_states + residual

        # Feedforward Layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super.__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Number of layers in the LLMs
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(GemmaDecoderLayer(config))
        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_cache,
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config

        # Multi-Head Attention
        # The hidden size that goes into the transformer
        # must be divisible by the number of attention heads evenly.
        # ---------------------------------------------------------
        # Gemma 7B
        # 3072 hidden size and 16 attention heads
        # So, each head has 3072 / 16 = 192 hidden size
        # ---------------------------------------------------------
        # Gemma 2B
        # 2048 hidden size and 8 attention heads
        # So, each head has 2048 / 8 = 256 hidden size
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings

        # Head dimensions
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        # Initialise the Tokenizer
        self.tokenizer = Tokenizer(config.tokenizer)

        # Initialise the Embedding layer
        self.embedder = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size,
            quant=config.quant,
        )

        # Initialise the Model
        self.model = GemmaModel(config=config)

        self.sampler = Sampler(vocab_size)


if __name__ == "__main__":
    config = GemmaConfig()
    gemma7b = GemmaForCausalLM(config)
