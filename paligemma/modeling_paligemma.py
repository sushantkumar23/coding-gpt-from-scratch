from dataclasses import dataclass
import torch
from typing import Tuple, Optional

import torch.nn as nn

from ..siglip.modeling_siglip import SiglipVisionConfig
from ..gemma.modeling_gemma import GemmaConfig


@dataclass
class PaliGemmaConfig:
    vision_config: SiglipVisionConfig
    text_config: GemmaConfig
    projection_dim: int = 2048
    hidden_size: int = 2048
    vocab_size: int = text_config.vocab_size
    pad_token_id: int = None


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        input_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        assert torch.all(attention_mask == 1), "the input cannnot be padded"

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_feature)

        input_embeds, attention_mark, position_ids = (
            self._merge_input_ids_with_image_features(
                input_embeds, image_features, attention_mask, kv_cache
            )
        )
