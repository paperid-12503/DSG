# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.builder import MODELS
from models.tcl.gumbel import gumbel_sigmoid
from models.tcl.modules import FeatureEncoder

from utils import get_logger
import us


@MODELS.register_module()
class Sim2Mask(nn.Module):
    def __init__(self, init_w=1.0, init_b=0.0, gumbel_tau=1.0, learnable=True):
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.gumbel_tau = gumbel_tau

        self.w1 = 6
        self.b1 = -4
        
        self.w2 = 10.0
        self.b2 = -2.5

    def forward(self, x, deterministic=False, is_train=True):
        if is_train:
            logits = x * self.w1 + self.b1 
        else:
            logits = x * self.w2 + self.b2

        soft_mask = torch.sigmoid(logits)
        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(logits.dtype)
        else:
            hard_mask = gumbel_sigmoid(logits, hard=True, tau=self.gumbel_tau)

        return hard_mask, soft_mask
    
    def wosigmoid(self, x, deterministic=False):
        logits = x * self.w + self.b
        return logits
    
    def extra_repr(self):
        return f'init_w={self.init_w}, init_b={self.init_b}, learnable={self.learnable}, gumbel_tau={self.gumbel_tau}'


class MaskerBackbone(nn.Module):
    """Masker image encoder backbone.
    """
    def __init__(self, clip_visual, freeze_idx):
        super().__init__()
        self.transformer = copy.deepcopy(clip_visual.transformer)
        self.transformer.resblocks = self.transformer.resblocks[freeze_idx:]

        for block in self.transformer.resblocks:
            if hasattr(block, "hook_handler"):
                block.hook_handler.remove()

        self.ln_post = copy.deepcopy(clip_visual.ln_post)
        self.proj = copy.deepcopy(clip_visual.proj)

        self.layers = len(self.transformer.resblocks)
        self.patch_size = clip_visual.patch_size

        self.output_dim = clip_visual.output_dim if self.proj is not None else clip_visual.width

    def forward(self, x, spatial=True, ignore_last_attn=True):
        if self.layers:
            x = self.transformer(x, ignore_last_attn=ignore_last_attn)

        x = x.permute(1, 0, 2)  # LND -> NLD

        if spatial:
            x = self.ln_post(x)
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class MaskerImageFeatureEncoder(FeatureEncoder):
    def __init__(self, backbone: nn.Module, decoder: nn.Module, ignore_last_attn: bool = True):
        super().__init__()
        self.ignore_last_attn = ignore_last_attn
        self.patch_size = backbone.patch_size
        self.backbone = backbone
        self.decoder = decoder

        for resblock in self.backbone.transformer.resblocks:
            resblock.hook_handler = resblock.register_forward_hook(self.hook)

    def _encode(self, image, image_feat, need_image_feat=False):
        H, W = image.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size

        x = self.backbone(image_feat, spatial=True, ignore_last_attn=self.ignore_last_attn)  # BLC
        x = rearrange(x[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
        x = self.decoder(x)

        return x


@MODELS.register_module()
class Masker(nn.Module):
    def __init__(self, backbone, decoder, image_proj, sim2mask, ignore_last_attn, **kwargs):
        super().__init__()
        self.ignore_last_attn = ignore_last_attn

        decoder["C"] = backbone.output_dim
        decoder = MODELS.build(decoder)
        decoder = nn.Sequential(OrderedDict([
            ("decoder", decoder),
            ("image_proj", image_proj)
        ]))

        self.image_encoder = MaskerImageFeatureEncoder(backbone, decoder, ignore_last_attn=ignore_last_attn)

        self.sim2mask = Sim2Mask(**sim2mask)

    def forward(self, image_emb, q_emb, deterministic=False, is_train=True):
        B = image_emb.size(0) # B * nouns

        H, W = image_emb.shape[2:]
        D = dist.get_world_size()

        # simmap [B, B*D, H, W] where D is #devices
        all_q_emb_norm = us.gather_cat(q_emb, grad=True, contiguous_grad=True)
        simmap = torch.einsum("bchw,nc->bnhw", image_emb, all_q_emb_norm)
        _, soft_mask = self.sim2mask(simmap, deterministic=deterministic, is_train=is_train) # bnhw

        # mask [B, B*D, H, W] where D is #devices
        # positive global label
        pos_indices = torch.arange(B, dtype=torch.long, device=image_emb.device) + B * dist.get_rank()

        offdiag = torch.ones(B, B*D, dtype=torch.bool, device=soft_mask.device)
        offdiag[torch.arange(B), pos_indices] = False

        soft_pos_mask = soft_mask[torch.arange(B), pos_indices].unsqueeze(1)

        masks = {
            # "pos": pos_mask,  # [B, 1, H, W]
            "soft_pos": soft_pos_mask,
            # "soft_neg": soft_neg_mask,
            "soft_all": soft_mask,  # [B, N, H, W]
        }

        return masks

    def get_mask_only(self, image_emb, text_emb, deterministic=False):
        B, C, H, W = image_emb.shape

        # simmap [B, B*D, H, W] where D is #devices
        text_emb = text_emb.view(B, -1, C)
        simmap = torch.einsum("bchw,bnc->bnhw", image_emb, text_emb)
        soft_mask = self.sim2mask.wosigmoid(simmap, deterministic=deterministic)
        return soft_mask

    @torch.no_grad()
    def forward_seg(self, image, image_feat, text_emb, deterministic=True, hard=False):
        """
        Make mask by 1:N matching

        Args:
            image [B, 3, H, W]
            image_feat [L, B, C]: CLIP features
            text_emb [N, C]
            deterministic (bool): deterministic inference flag for gumbel noise
            hard (bool): decide hard or soft returning segmentation mask.
                Note that soft mask is required for proper evaluation

        Return:
            mask [B, N, H', W'] (H' and W' are downsampled H/W)
        """
        image_emb = self.image_encoder(image, image_feat)  # [BCHW]

        image_emb = us.normalize(image_emb, dim=1)  # BCHW
        text_emb = us.normalize(text_emb, dim=-1)  # NC

        simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_emb)

        hard_mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic, is_train=False)
        mask = hard_mask if hard else soft_mask

        return mask, simmap
