# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Add upscale_pred flag
# - Update debug_output system
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from mmseg.ops import resize
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from .projection import ProjectionHead
from .module_helper import ModuleHelper

from .layers import SEBlock, SABlock
from fightingcv_attention.attention.SEAttention import SEAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img

class BottleneckPad(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BottleneckPad, self).__init__()
        affine_par = True
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module2048(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module2048, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """

    def __init__(self, tasks=['S','D'], input_channels=2048, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = tasks
        layers = {}
        conv_out = {}
        NUM_OUTPUT = {"S": 19, "D": 1, "D_src": 1}

        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                     stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = BottleneckPad(input_channels, intermediate_channels // 4, downsample=downsample)
            bottleneck2 = BottleneckPad(intermediate_channels, intermediate_channels // 4, downsample=None)
            if task == "S":
                conv_out_ = Classifier_Module2048([6, 12, 18, 24], [6, 12, 18, 24], NUM_OUTPUT[task])
            else:
                conv_out_ = nn.Conv2d(intermediate_channels, NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_
        conv_out["D_src"] = nn.Conv2d(intermediate_channels, NUM_OUTPUT["D"], 1)
        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}

        for task in self.tasks:
            out['features_%s' % (task)] = self.layers[task](x)
        out["D_src"] = self.conv_out["D_src"](out['features_D'])
        out["D"] = self.conv_out["D"](out['features_D'])
        out["S"] = self.conv_out["S"](x)
        return out


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}

        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' % (a)]) for a in self.auxilary_tasks if a != t} for
                    t in self.tasks}
        out = {t: x['features_%s' % (t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in
               self.tasks}
        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        vec = y.expand_as(x)
        return x * y.expand_as(x), vec

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W

class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x
class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

@SEGMENTORS.register_module()
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scales=[1],
                 hr_crop_size=None,
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop

        # Add self.cls_head and self.proj_head
        # extra added layers
        self.num_classes = 19
        self.proj_dim = 256
        in_channels = 1024  # 64 + 128 + 320 + 512
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type='torchbn'),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)

        # Add Corda
        self.tasks = ['S', 'D']
        self.auxilary_tasks = ['S', 'D']
        NUM_OUTPUT = {"S": 19, "D": 1, "D_src": 1}
        self.channels = 2048
        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.auxilary_tasks, self.channels)
        # Multi-modal distillation
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)
        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks + ["D_src"]:
            heads[task] = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24], NUM_OUTPUT[task])
        self.heads = nn.ModuleDict(heads)

        # Transformer
        self.depthnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1)
        self.segnorm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1)
        # patch_embed
        embed_dims = [64, 128, 256, 512]
        img_size=224
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=224,
            patch_size=7,
            stride=4,
            in_chans=512,
            embed_dim=embed_dims[0])
        depths = [3, 4, 6, 3]
        cur = 0
        dpr = [
            x.item() for x in torch.linspace(0, 0.1, sum(depths))
        ]  # stochastic depth decay rule
        self.block1 = nn.ModuleList([
            Block(
                dim=64,
                num_heads=1,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[cur + i],
                norm_layer=nn.LayerNorm,
                sr_ratio=8) for i in range(3)
        ])
        norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(embed_dims[0])

        self.block2 = nn.ModuleList([
            Block(
                dim=64,
                num_heads=1,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[cur + i],
                norm_layer=nn.LayerNorm,
                sr_ratio=8) for i in range(3)
        ])
        norm_layer = nn.LayerNorm
        self.norm2 = norm_layer(embed_dims[0])

        self.block3 = nn.ModuleList([
            Block(
                dim=64,
                num_heads=1,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[cur + i],
                norm_layer=nn.LayerNorm,
                sr_ratio=8) for i in range(3)
        ])
        norm_layer = nn.LayerNorm
        self.norm3 = norm_layer(embed_dims[0])

        self.block4 = nn.ModuleList([
            Block(
                dim=64,
                num_heads=1,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[cur + i],
                norm_layer=nn.LayerNorm,
                sr_ratio=8) for i in range(3)
        ])
        norm_layer = nn.LayerNorm
        self.norm4 = norm_layer(embed_dims[0])

        self.block5 = nn.ModuleList([
            Block(
                dim=64,
                num_heads=1,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[cur + i],
                norm_layer=nn.LayerNorm,
                sr_ratio=8) for i in range(3)
        ])
        norm_layer = nn.LayerNorm
        self.norm5 = norm_layer(embed_dims[0])

        # Depth Selection
        self.selection = nn.Conv2d(
            64,
            1,
            kernel_size=1,
            stride=1,)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def extract_unscaled_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def generate_pseudo_label(self, img, img_metas):
        self.update_debug_state()
        out = self.encode_decode(img, img_metas)
        if self.debug:
            self.debug_output = self.decode_head.debug_output
        return out

    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = self.resize(img, s)
            if i >= 1 and self.hr_slide_inference:
                mres_feats.append(self.extract_slide_feat(scaled_img))
            else:
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
        out = self._decode_head_forward_test(mres_feats, img_metas)
        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _forward_train_features(self, img):
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        return mres_feats, prob_vis

    def forward_sum(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward_sum1(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward_sum2(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_sum3(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_sum4(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block5):
            x = blk(x, H, W)
        x = self.norm5(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward_train(self,
                      stage,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      depth,
                      depth_target,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()

        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        # Calculate emb
        x_hr = mres_feats[1]
        # low resolution features
        x_lr = mres_feats[0]
        _, _, h, w = x_hr[0].size()

        feat1 = x_hr[0]
        feat2 = F.interpolate(x_hr[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x_hr[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x_hr[3], size=(h, w), mode="bilinear", align_corners=True)

        # Corda features
        feat1_corda = F.interpolate(x_hr[0], size=(65, 65), mode="bilinear", align_corners=True)
        feat2_corda = F.interpolate(x_hr[1], size=(65, 65), mode="bilinear", align_corners=True)
        feat3_corda = F.interpolate(x_hr[2], size=(65, 65), mode="bilinear", align_corners=True)
        feat4_corda = F.interpolate(x_hr[3], size=(65, 65), mode="bilinear", align_corners=True)
        feats_corda = torch.cat([feat1_corda, feat2_corda, feat3_corda, feat4_corda], 1)
        feats_corda = torch.cat([feats_corda, feats_corda], 1)

        out = {}
        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(feats_corda)

        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = x[task]
        out["initial_D_src"] = x["D_src"]

        # Depth Adaptative Selection
        depth_info = x['features_D']
        depth_info = F.interpolate(depth_info, size=(512, 512), mode="bilinear", align_corners=True)
        seg_info = x['features_S']
        seg_info = F.interpolate(seg_info, size=(512, 512), mode="bilinear", align_corners=True)
        depth_info = self.depthnorm(depth_info)
        seg_info = self.segnorm(seg_info)
        sum_info = torch.cat([depth_info, seg_info], dim=1)

        # SENet
        # input = sum_info
        # se = SEAttention(channel=256, reduction=8)
        # se.cuda()
        # output, vec = se(input)
        #
        # depth_info = depth_info * vec
        # x['feature_D'] = depth_info

        # Segformer
        ex = self.forward_sum(sum_info)
        ex = F.interpolate(ex, size=(65, 65), mode="bilinear", align_corners=True)
        ex = self.selection(ex)
        ex = torch.sigmoid(ex)

        selected_depth_info = x['features_D']
        selected_depth_info = selected_depth_info * ex
        x['feature_D'] = selected_depth_info


        # also change visual features
        selected_seg_info = x['features_S']
        selected_seg_info = selected_seg_info * ex
        x['feature_S'] = selected_seg_info

        # Information Communication 1
        df1 = F.interpolate(selected_depth_info, size=(256, 256), mode="bilinear", align_corners=True)
        vf1 = F.interpolate(selected_seg_info, size=(256, 256), mode="bilinear", align_corners=True)
        sum_info_1 = torch.cat([df1, vf1], dim=1)

        ex = self.forward_sum1(sum_info_1)
        ex = F.interpolate(ex, size=(65, 65), mode="bilinear", align_corners=True)
        ex = self.selection(ex)
        ex = torch.sigmoid(ex)

        selected_depth_info = x['features_D']
        selected_depth_info = selected_depth_info * ex
        x['feature_D'] = selected_depth_info
        selected_seg_info = x['features_S']
        selected_seg_info = selected_seg_info * ex
        x['feature_S'] = selected_seg_info

        # Information Communication 2
        df2 = F.interpolate(selected_depth_info, size=(256, 256), mode="bilinear", align_corners=True)
        vf2 = F.interpolate(selected_seg_info, size=(256, 256), mode="bilinear", align_corners=True)
        sum_info_2 = torch.cat([df2, vf2], dim=1)

        ex = self.forward_sum2(sum_info_2)
        ex = F.interpolate(ex, size=(65, 65), mode="bilinear", align_corners=True)
        ex = self.selection(ex)
        ex = torch.sigmoid(ex)

        selected_depth_info = x['features_D']
        selected_depth_info = selected_depth_info * ex
        x['feature_D'] = selected_depth_info
        selected_seg_info = x['features_S']
        selected_seg_info = selected_seg_info * ex
        x['feature_S'] = selected_seg_info

        # Information Communication 3
        df3 = F.interpolate(selected_depth_info, size=(256, 256), mode="bilinear", align_corners=True)
        vf3 = F.interpolate(selected_seg_info, size=(256, 256), mode="bilinear", align_corners=True)
        sum_info_3 = torch.cat([df3, vf3], dim=1)

        ex = self.forward_sum3(sum_info_3)
        ex = F.interpolate(ex, size=(65, 65), mode="bilinear", align_corners=True)
        ex = self.selection(ex)
        ex = torch.sigmoid(ex)

        selected_depth_info = x['features_D']
        selected_depth_info = selected_depth_info * ex
        x['feature_D'] = selected_depth_info
        selected_seg_info = x['features_S']
        selected_seg_info = selected_seg_info * ex
        x['feature_S'] = selected_seg_info

        # Information Communication 4
        df4 = F.interpolate(selected_depth_info, size=(256, 256), mode="bilinear", align_corners=True)
        vf4 = F.interpolate(selected_seg_info, size=(256, 256), mode="bilinear", align_corners=True)
        sum_info_4 = torch.cat([df4, vf4], dim=1)

        ex = self.forward_sum4(sum_info_4)
        ex = F.interpolate(ex, size=(65, 65), mode="bilinear", align_corners=True)
        ex = self.selection(ex)
        ex = torch.sigmoid(ex)

        selected_depth_info = x['features_D']
        selected_depth_info = selected_depth_info * ex
        x['feature_D'] = selected_depth_info
        selected_seg_info = x['features_S']
        selected_seg_info = selected_seg_info * ex
        x['feature_S'] = selected_seg_info

        # Refine features through multi-modal distillation
        # x_refine = self.multi_modal_distillation(x)

        x['D'] = x['feature_D']
        x['S'] = x['feature_S']

        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = self.heads[task](x[task])
            if task == "D":
                out["D_src"] = self.heads["D_src"](x[task])



        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_class = 0

        emb = 0

        feature_outputs = {'seg': out_class, 'embed': emb, 'corda': out, 'depth':depth, 'depth_target':depth_target}

        loss_decode = self._decode_head_forward_train(mres_feats, feature_outputs, stage, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      return_logits)
        losses.update(loss_decode)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
        self.local_iter += 1
        return losses

    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}
