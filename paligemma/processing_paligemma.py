from typing import List, Tuple
from PIL import Image
import numpy as np
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class PaligemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image]):

        # Resizing
        images = [
            image.resize(
                size=(self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC,
            )
            for image in images
        ]
        images = [np.array(image) for image in images]

        # Rescaling
        images = [(image / 255.0).astype(np.float32) for image in images]

        # Normalizing
        mean = np.array(IMAGENET_STANDARD_MEAN).astype(np.float32)
        std = np.array(IMAGENET_STANDARD_STD).astype(np.float32)
        images = [(image - mean) / std for image in images]
        images = [image.transpose(2, 0, 1) for image in images]

        pixel_values = np.stack(images, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            f"{self.IMAGE_TOKEN * self.image_seq_len}{self.tokenizer.bos_token}{prompt}\n"
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding="longest", truncation=True
        )

        return_data = {
            "pixel_values": pixel_values,
            **inputs,
        }

        return return_data
