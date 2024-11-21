import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

from models.builder import MODELS

from models.tcl.noun_decomposition import ImageDecomposition
from models.tcl.decoders import TextDecoder
from sclip import tokenize
from models.tcl.bg_block import *


@MODELS.register_module()
class ImageTextCoDecomposition(ImageDecomposition):
    def __init__(
        self,
        w_hcl,
        w_tseg,
        # w_ce,
        # use_word_highlighting_prompt,
        train_with_bg = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.eps = 0.0000001
        self.topk = 1
        self.w_hcl = w_hcl
        self.w_tseg = w_tseg

        self.bce_loss = nn.BCELoss()
        self.bg_bce =  nn.BCEWithLogitsLoss()
        self.train_with_bg = train_with_bg
        if train_with_bg:
            self.bg_block = Aggregator()

    def forward(self, image, category, mask_gt, text, use_pamr=False):
        """
            image:    <B, 3, 224, 224>
            category: <B, n, 512>
            mask:     <B, n, 224, 224>
            text:     Boolean
        """

        b, n, H, W = mask_gt.shape
        category = category / category.norm(dim=-1, keepdim=True)
        q_embeddings = category.permute(1, 0, 2).reshape(b*n, -1) # <b, n, dim>
        ret = {}  

        if not text:
            embs = self.patch_image_embedding(image, category, mask_gt)
            embs = embs['emb_type'].permute(1, 0, 2).reshape(b*n, -1)

            q_embeddings = (q_embeddings + embs) / 2
        
        decoded_feat_map, feats = self.decode_feature_map(image, need_feats=True)
        decoded_feat_map = torch.cat([decoded_feat_map] * n, dim=0)
        image = torch.cat([image] * n, dim=0)
        masks = self.masker(decoded_feat_map, q_embeddings, is_train=True)
        # masks = self.masker(decoded_feat_map, embs, is_train=True)

        # bgloss
        if self.train_with_bg:
            ret_new = self.background_cls(category, mask_gt, feats, justtest=q_embeddings)
            ret.update(ret_new)

        mask_pos = torch.cat([masks['soft_all'][i, i:i+1, :, :] for i in range(masks['soft_all'].shape[0])], dim=0)
        pred = F.interpolate(masks['soft_pos'], size=(H, W), mode='bilinear', align_corners=False).squeeze(dim=1).float()
        mask_gt = mask_gt.permute(1, 0, 2, 3).reshape(mask_pos.shape[0], H, W).float()
        with autocast(enabled=False):
            ret['bce_loss'] = self.bce_loss(pred, mask_gt) * 2.5
        ret['dice_loss'] = self.dice_loss(pred, mask_gt) * 0.9

        if text is not None:
            new_ret, _ = self.cal_iseg_loss_mask(
                image,
                mask_gt,
                decoded_feat_map,
                q_embeddings,
            )
            ret.update(new_ret)
        
        if use_pamr:
            masks["soft_pos"] = self.apply_pamr(image, masks["soft_pos"])

        records = {
            "image": image,
            "masks": masks,
        }

        return ret, records

    def masked_pool(self, spatial_image_emb, mask, eps=1e-6):
        """
        Average pool spatial_image_emb with mask

        Args:
            spatial_image_emb [BCHW]: spatial embedding
            mask [BNHW]: hard or soft mask

        Return:
            image_emb [BNC] : mask-pooled tensor
        """
        mask_sum = mask.sum((2,3), keepdim=True)  # [BN11]
        weight = mask / (mask_sum + eps)
        masked_image_emb = torch.einsum("bchw,bnhw->bnc", spatial_image_emb, weight)  # [BNC]

        return masked_image_emb
    
    def patch_image_embedding(self, image, category, mask_gt):
        """
        Args:
            image: <B, 3, 224, 224>
            category: <B, n, 512>
            mask_gt: <B, n, 224, 224>
        """
        with torch.no_grad():
            B, n, H, W = mask_gt.shape
            frozen_embedding = self.clip_image_encoder(image)
            frozen_embedding = self.clip_image_encoder.clip_proj(frozen_embedding).unsqueeze(1).repeat(1, n, 1, 1, 1)
            mask_gt_14 = F.interpolate(mask_gt, size=frozen_embedding.shape[-2:], mode='nearest').unsqueeze(dim=2)
            zeros = mask_gt_14.sum(dim=(2,3,4))

            emb_type1 = (frozen_embedding * mask_gt_14).sum(dim=(3, 4)) / (mask_gt_14.sum(dim=(3, 4)) + self.eps)
            emb_type1[zeros==0, :] = category[zeros==0, :].float()
            emb_type1 = emb_type1 / emb_type1.norm(dim=-1, keepdim=True)

            
            # emb_copy = (frozen_embedding * mask_gt_14).view(B, n, 512, -1)
            # similarity = torch.einsum('bncx,bnc->bnx', emb_copy, category)  # (64, 2, 196)
            # _, topk_indices = torch.topk(similarity, self.topk, dim=-1)  # (64, 2, topk)
            # emb_type2 = torch.gather(emb_copy, 3, topk_indices.unsqueeze(2).expand(-1, -1, 512, -1))  # (64, 2, 512, topk)
            # emb_type2[zeros < self.topk, :, :] = category[zeros < self.topk, :].unsqueeze(dim=-1).repeat(1, 1, self.topk).half()
            # emb_type2 = emb_type2.mean(dim=-1)
            # emb_type2 = emb_type2 / emb_type2.norm(dim=-1, keepdim=True)

            
            # similarity_sum = similarity.sum(dim=-1, keepdim=True)
            # weight = similarity / (similarity_sum + self.eps)      
            # emb_type3 = torch.einsum("bncx,bnx->bnc", emb_copy, weight)  
            # emb_type3[zeros==0, :] = category[zeros==0, :].half()
            # emb_type3 = emb_type3 / emb_type3.norm(dim=-1, keepdim=True)

            # emb_typeall = emb_type2 + emb_type3
            # emb_typeall = emb_typeall / emb_typeall.norm(dim=-1, keepdim=True)

        emb = {"emb_type": emb_type1}
        # emb = {"emb_type1": emb_type1.half(), "emb_type2": emb_type2, "emb_type3": emb_type3}
        return emb
    
    def background_cls(self, category, mask_gt, decoded_feat_map, justtest=None):
        decoded_feat_map = rearrange(decoded_feat_map, '(h w) b c -> b c h w', h=14, w=14)
        category = rearrange(category, 'b n c -> (b n) c')
        GT = (mask_gt.sum(dim=1)>0).type(torch.int8)
        pred = self.bg_block(decoded_feat_map, category, justtest)
        pred = F.interpolate(pred, size=GT.shape[-2:], mode='bilinear', align_corners=False).squeeze(dim=1)
        
        ret_new = {"bg_bce_loss": self.bg_bce(pred, GT.float())}
        return ret_new