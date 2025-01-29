# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPVisionConfig,
    Dinov2Model,
    Dinov2Config,
)

def split_tiles(embeds, num_split):
    B, C, H, W = embeds.shape
    out = []
    for x in embeds:  # x shape: [C, H, W]
        x = x.unsqueeze(0)  # shape: [1, C, H, W]
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] 
                           for i in range(num_split) 
                           for j in range(num_split)], dim=0)    
        out.append(x_split)
        print("x_split", x_split.shape)
    
    x_split = torch.stack(out, dim=0)  # Final shape: [B, num_split*num_split, C, h, w]
    
    return x_split

# matteo's ipadapter tiling code
def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        print("embeds", embeds.shape)
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])
        
        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)
    
    out = torch.cat(out, dim=0)

    return out

class ImageEncoder(nn.Module):
    def __init__(
        self,
        version=None,
        config=None,
        use_cls_token=True,
        image_size=224,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            self.model = self.MODEL_CLASS.from_pretrained(version)
        else:
            self.model = self.MODEL_CLASS(self.MODEL_CONFIG_CLASS.from_dict(config))
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_cls_token = use_cls_token
        self.size = image_size // 14
        self.num_patches = (image_size // 14) ** 2
        if self.use_cls_token:
            self.num_patches += 1

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

    def forward(self, image, mask=None, value_range=(-1, 1), tiles = 1, ratio = 0.8):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.model.device, dtype=self.model.dtype)
        
        if mask is not None:
            mask = mask.to(image)
            image = image * mask

        image = self.transform(image)
        
        last_hidden_state = self.model(image).last_hidden_state
    
        if tiles > 1:
            hidden_state = None
            image_split = split_tiles(image, tiles)
            for i in image_split:
                i = self.transform(i)
                if hidden_state is None:
                    hidden_state = self.model(i).last_hidden_state
                else:
                    hidden_state = torch.cat([hidden_state, self.model(i).last_hidden_state], dim=0)
            hidden_state = merge_hiddenstates(hidden_state, tiles)
            last_hidden_state = last_hidden_state*ratio + hidden_state * (1-ratio)
        
        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]
        

        return last_hidden_state

    def unconditional_embedding(self, batch_size):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        zero = torch.zeros(
            batch_size,
            self.num_patches,
            self.model.config.hidden_size,
            device=device,
            dtype=dtype,
        )

        return zero


class CLIPImageEncoder(ImageEncoder):
    MODEL_CLASS = CLIPVisionModelWithProjection
    MODEL_CONFIG_CLASS = CLIPVisionConfig
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]


class DinoImageEncoder(ImageEncoder):
    MODEL_CLASS = Dinov2Model
    MODEL_CONFIG_CLASS = Dinov2Config
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


def build_image_encoder(config):
    if config['type'] == 'CLIPImageEncoder':
        return CLIPImageEncoder(**config['kwargs'])
    elif config['type'] == 'DinoImageEncoder':
        return DinoImageEncoder(**config['kwargs'])
    else:
        raise ValueError(f'Unknown image encoder type: {config["type"]}')


class DualImageEncoder(nn.Module):
    def __init__(
        self,
        main_image_encoder,
        additional_image_encoder,
    ):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        self.additional_image_encoder = build_image_encoder(additional_image_encoder)

    def forward(self, image, mask=None):
        outputs = {
            'main': self.main_image_encoder(image, mask=mask),
            'additional': self.additional_image_encoder(image, mask=mask),
        }
        return outputs

    def unconditional_embedding(self, batch_size):
        outputs = {
            'main': self.main_image_encoder.unconditional_embedding(batch_size),
            'additional': self.additional_image_encoder.unconditional_embedding(batch_size),
        }
        return outputs


class SingleImageEncoder(nn.Module):
    def __init__(
        self,
        main_image_encoder,
    ):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)

    def forward(self, image, mask=None, tiles = 1, ratio = 0.8):
        outputs = {
            'main': self.main_image_encoder(image, mask=mask, tiles = tiles, ratio = ratio),
        }
        return outputs

    def unconditional_embedding(self, batch_size):
        outputs = {
            'main': self.main_image_encoder.unconditional_embedding(batch_size),
        }
        return outputs
