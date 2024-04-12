# model.py
import os
from typing import Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor


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
    tokenizer: Optional[str] = 'tokenizer/tokenizer.model'



def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384
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


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False
            )
        self.quant = quant


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        quant: bool
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0


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
            quant=config.quant
        )



class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super.__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Number of layers in the LLMs
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(GemmaDecoderLayer(config))
        



class GemmaForCausalLM(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config
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
            quant=config.quant
        )

        # Initialise the Model
        self.model = GemmaModel(config=config)



if __name__ == "__main__":
    config = GemmaConfig()
    gemma7b = GemmaForCausalLM(config)