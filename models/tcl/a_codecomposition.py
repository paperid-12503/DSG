import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast

from models.builder import MODELS
from models.tcl.mi import InfoNCE
import us
import torch.distributed as dist

from models.tcl.noun_decomposition import ImageDecomposition
from models.tcl.decoders import TextDecoder
from sclip import tokenize

@MODELS.register_module()
class ImageTextCoDecomposition(ImageDecomposition):
    def __init__(
        self,
        w_hcl,
        w_tseg,
        use_word_highlighting_prompt,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.w_hcl = w_hcl
        self.w_tseg = w_tseg

        self.bce_loss = nn.BCELoss()
        
    @torch.no_grad()
    def encode_text_features(self, caption):
        text_token_ids = tokenize(caption, context_length=77, truncate=True)
        text_token_ids = text_token_ids.cuda()

        _, text_hidden_embs = self.frozen_clip.encode_text(text_token_ids, True)

        text_tokens, text_indices = self.frozen_clip.get_word_tokens(
            text_token_ids)
        text_tokens = text_tokens.permute(1, 0, 2)
        return {
            "text_hidden_embs": text_hidden_embs,
            "text_tokens": text_tokens,
            "text_indices": text_indices,
        }


    def forward(self, image, category, mask_gt, caption, use_pamr=False):
        num_nouns = len(category)
        all_nouns = sum((noun_list for noun_list in category), [])

        ret = {}  # losses + logs

        decoded_feat_map = self.decode_feature_map(image)
        decoded_feat_map = torch.cat([decoded_feat_map] * num_nouns, dim=0)
        image = torch.cat([image] * num_nouns, dim=0)

        # Build noun embeddings
        noun_embs = self.clip_text_encoder(all_nouns)
        ret["kg_loss"] = self.w_kg * self.cal_kg_loss(noun_embs, all_nouns)

        masks = self.masker(decoded_feat_map, noun_embs)

        mask_pos = torch.cat([masks['soft_all'][i, i:i+1, :, :] for i in range(masks['soft_all'].shape[0])], dim=0)
        mask_resize_pos = F.interpolate(mask_pos.unsqueeze(0), size=mask_gt.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        mask_gt = mask_gt.permute(1, 0, 2, 3).reshape(mask_pos.shape[0], mask_gt.shape[-2], mask_gt.shape[-1])
        # with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        with autocast(enabled=False):
            ret['bce_loss'] = self.bce_loss(mask_resize_pos.float(), mask_gt.float()) * 2

        new_ret, fg_image_emb = self.cal_iseg_loss(
            image,
            masks,
            decoded_feat_map,
            noun_embs,
        )

        ret.update(new_ret)

        if use_pamr:
            masks["soft_pos"] = self.apply_pamr(image, masks["soft_pos"])

        records = {
            "image": image,
            "masks": masks,
        }

        return ret, records
