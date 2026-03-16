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

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPVisionConfig,
    Dinov2Model,
    Dinov2Config,
)
import logging
log = logging.getLogger(__name__)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


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
        self.has_guidance_embed = kwargs.get('has_guidance_embed', False)

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

        #MV
        self.view_num = 4
        pos = np.arange(self.view_num, dtype=np.float32)
        view_embedding = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(self.model.config.hidden_size, pos)).float()
        view_embedding = view_embedding.unsqueeze(1).repeat(1, self.num_patches, 1)
        self.view_embed = view_embedding.unsqueeze(0)
        
        self.view2idx = {
            'front': 0,
            'left': 1,
            'back': 2,
            'right': 3
        }

    def forward(self, image, mask=None, value_range=(-1, 1), view_dict=None):
        if view_dict is None:
            self.view_num = 1
            if value_range is not None:
                low, high = value_range
                image = (image - low) / (high - low)

            image = image.to(self.model.device, dtype=self.model.dtype)
            print("image shape", image.shape)
            
            if mask is not None:
                mask = mask.to(image)
                image = image * mask
            supported_sizes = [518, 530]
            if (image.shape[2] not in supported_sizes or image.shape[3] not in supported_sizes) and not self.has_guidance_embed:
                log.info(f'Image shape {image.shape} not supported. Resizing to 518x518')
                inputs = self.transform(image)
            else:
                inputs = image
            outputs = self.model(inputs)

            last_hidden_state = outputs.last_hidden_state
            if not self.use_cls_token:
                last_hidden_state = last_hidden_state[:, 1:, :]

            return last_hidden_state
        else:
            images = []
            view_indexes = []
            supported_sizes = [518, 530]

            for view_tag, view_image in view_dict.items():
                if view_image is not None:
                    if view_image.shape[1] == 4:
                        log.info(f'{view_tag} view image has alpha channel, using it to set background to black')
                        rgb = view_image[:, :3, :, :]
                        alpha = view_image[:, 3:4, :, :]
                        view_image = rgb * alpha
                    else:
                        log.warning(f"{view_tag} view image has no alpha channel, make sure the background is already black")
                    view_image = view_image.to(self.model.device, dtype=self.model.dtype)
                    if (view_image.shape[2] not in supported_sizes or view_image.shape[3] not in supported_sizes):
                        log.info(f'view_image shape {view_image.shape} not supported. Resizing to 518x518')
                        view_image = self.transform(view_image)
                    images.append(view_image)
                    view_indexes.append(self.view2idx[view_tag])

            image_tensors = torch.cat(images, 0)

            outputs = self.model(image_tensors)

            self.view_num = len(images)

            last_hidden_state = outputs.last_hidden_state
            last_hidden_state = last_hidden_state.view(
                self.view_num, 
                -1,
                last_hidden_state.shape[-1]
            )
            view_embedding = self.view_embed.to(last_hidden_state.dtype).to(last_hidden_state.device)
            if self.view_num != 4:
                view_embeddings = []
                for idx in range(len(view_indexes)):
                    view_embeddings.append(self.view_embed[:, idx, ...])
                view_embedding = torch.cat(view_embeddings, 0).to(last_hidden_state.dtype).to(last_hidden_state.device)

            last_hidden_state = last_hidden_state + view_embedding
            last_hidden_state = last_hidden_state.view(-1, last_hidden_state.shape[-1])
            return last_hidden_state.unsqueeze(0)

    def unconditional_embedding(self, batch_size):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        zero = torch.zeros(
            batch_size,
            self.num_patches * self.view_num,
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

    def forward(self, image=None, mask=None, view_dict=None):
        outputs = {
            'main': self.main_image_encoder(image=image, mask=mask, view_dict=view_dict),
        }
        return outputs

    def unconditional_embedding(self, batch_size):
        outputs = {
            'main': self.main_image_encoder.unconditional_embedding(batch_size),
        }
        return outputs