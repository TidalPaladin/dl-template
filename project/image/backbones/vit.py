#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from functools import partial
from typing import Any, List, TypeVar, Union, cast, Tuple, Optional, Type, TypeAlias
from registry import Registry

import torch.nn as nn
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, requires, _TIMM_AVAILABLE
from flash.core.utilities.providers import _TORCHVISION, _TIMM
from flash.core.utilities.url_error import catch_url_error
from torch import Tensor

from ...core.adapter import BackboneAdapter, SequentialList, Backbone


if _TIMM_AVAILABLE:
    import timm
    from timm.models.layers import PatchEmbed
    from timm.models.vision_transformer import VisionTransformer
    from timm.models.layers.helpers import to_2tuple

    VIT_MODELS = timm.models.list_models("vit_*")

T = TypeVar("T", Tensor, List[Tensor])

class ViTStem(nn.Module):

    def __init__(
        self, 
        patch_embed: nn.Module, 
        pos_embed: nn.Parameter,
        cls_token: Optional[nn.Parameter] = None,
        pos_drop: nn.Dropout = nn.Dropout(0)
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.cls_token = cls_token
        self.pos_drop = pos_drop

    @property
    def embed_dim(self) -> int:
        assert isinstance(self.patch_embed, PatchEmbed)
        return self.patch_embed.proj.out_channels

    @classmethod
    def from_vit( cls,  vit: nn.Module) -> "ViTStem":
        if not isinstance(vit, VisionTransformer):
            raise TypeError(type(vit))
        return cls(vit.patch_embed, vit.pos_embed, vit.cls_token, vit.pos_drop)

    def forward_pos_embed(self, x: Tensor) -> Tensor:
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_patch(self, x: Tensor) -> Tensor:
        return self.patch_embed(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_patch(x)
        x = self.forward_pos_embed(x)
        return x


class ViTAdapter(BackboneAdapter[List[Tensor]]):

    @requires("timm")
    @classmethod
    def load_model(
        cls, 
        model_name: str, 
        pretrained: bool = False, 
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_chans: Optional[int] = None,
        class_token: Optional[bool] = None,
    ) -> nn.Module:
        model = timm.models.create_model(model_name, bool(pretrained))
        if any(v is not None for v in (img_size, patch_size, in_chans, class_token)):
            model = cls.adjust_stem(model, img_size, patch_size, in_chans, class_token)
        return model

    @classmethod
    def extract_stem(cls, model: nn.Module) -> nn.Module:
        return ViTStem.from_vit(model)

    @classmethod
    def extract_body(cls, model: nn.Module) -> nn.Module:
        return cast(nn.Module, model.blocks)

    @classmethod
    def extract_num_features(cls, model_name: str, model: nn.Module) -> int:
        return cast(int, model.num_features)

    @classmethod
    def adjust_stem(
        cls, 
        vit: nn.Module,
        img_size: Optional[Union[int, Tuple[int, int]]],
        patch_size: Optional[Union[int, Tuple[int, int]]],
        in_chans: Optional[int] = None,
        class_token: Optional[bool] = None,
    ) -> "VisionTransformer":
        if not isinstance(vit, VisionTransformer):
            raise TypeError(type(vit))

        # compute new size properties
        img_size = cast(Tuple[int, int], to_2tuple(img_size or vit.patch_embed.patch_size * vit.patch_embed.num_patches))
        patch_size = cast(Tuple[int, int], to_2tuple(patch_size or vit.patch_embed.patch_size))
        grid_size = tuple(i // p for i, p in zip(img_size, patch_size))
        num_patches = grid_size[0] * grid_size[1]
        in_chans = in_chans or vit.patch_embed.proj.in_channels
        embed_dim = vit.patch_embed.proj.out_channels
        class_token = class_token if class_token is not None else bool(vit.num_prefix_tokens)
        embed_len = num_patches + 1 if class_token else num_patches

        # determine which layers must be updated
        needs_new_patch_embed = (
            in_chans != vit.patch_embed.proj.in_channels
            or
            patch_size != vit.patch_embed.patch_size
        )
        needs_new_pos_emb = (
            num_patches != vit.patch_embed.num_patches
            or
            class_token != bool(vit.num_prefix_tokens)
        )

        patch_embed = (
            PatchEmbed(img_size, patch_size, in_chans, embed_dim)
            if needs_new_patch_embed else vit.patch_embed
        )
        pos_embed = (
            nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
            if needs_new_patch_embed else vit.pos_embed
        )

        vit.patch_embed = patch_embed
        vit.pos_embed = pos_embed
        if vit.cls_token is not None and not class_token:
            vit.cls_token = None
        return vit


def register_vit_backbones(register: Registry):
    for model_name in VIT_MODELS:
        register(
            fn=catch_url_error(partial(ViTAdapter.load_backbone, model_name=model_name, img_size=196)),
            name=model_name,
        )
