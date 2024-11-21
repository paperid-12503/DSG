import torch.nn as nn
import math
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class Mlp(nn.Module):
    """ 
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1
    
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.feature_map = elu_feature_map
        self.eps = 1e-6

    def forward(self, query, x, guidance):
        """ 
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        B, N, H, W, C = query.shape
        q = self.q(rearrange(query, 'B N H W C -> (B H W) N C'))

        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        Q = self.feature_map(q)
        K = self.feature_map(k)

        v_length = v.size(1)
        v = v / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, v)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        out = queried_values.contiguous().mean(dim=1)
        out = rearrange(out, 'B H D -> B (H D)')
        out = rearrange(out, '(B H W) C -> B H W C', B=B, H=H, W=W)
        return out

class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, pooling_size=(4, 4), pad_len=128, l_tokens=2) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim + guidance_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.l_tokens = l_tokens
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
        self.learnable_tokes = nn.Parameter(torch.zeros(1, self.l_tokens, 1, 1, hidden_dim))
        self.learnable_tokes.requires_grad_(True)

    def forward(self, x, guidance, query):

        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
            query: B, C, H, W
        """

        B, C, T, H, W = x.size()
        if guidance.ndim == 2:
            guidance = guidance.unsqueeze(dim=0).expand(B, -1, -1)

        query = query.permute(0, 2, 3, 1).unsqueeze(dim=1).expand(-1, self.l_tokens, -1, -1, -1) # B,T,H,W,C1
        query = torch.cat([self.learnable_tokes.expand(B, -1, H, W, -1), query], dim=-1)
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H, W=W)
                x = torch.cat([x, padding_tokens], dim=2)

        x = rearrange(x, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H, W=W)

        # query = query + self.attention(self.norm1(query), self.norm2(x), guidance) # Attention
        query = self.attention(self.norm1(query), self.norm2(x), guidance) # Attention
        query = query + self.MLP(self.norm3(query)) # MLP
        query = rearrange(query, 'B H W C -> B C H W') 

        return query

class Spatialtransformer(nn.Module):
    r""" 
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, appearance_guidance, mask=None):
        """
        Args:
            x: input features with shape of (B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, C, T, H, W = x.shape

        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
            x = torch.cat([x, appearance_guidance], dim=-1)

        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            attn = self.softmax(attn) + mask
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x

class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, pooling_size=(5, 5), pad_len=256) -> None:
        super().__init__()
        self.spatial = Spatialtransformer(hidden_dim, appearance_guidance, nheads)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
            feats_map: B, C, H, W
        """
        x = self.spatial(x, appearance_guidance)
        x = self.attention(x, text_guidance, appearance_guidance)
        return x
    
class Aggregator(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        num_layers=1,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(7, 7),
        feature_resolution=(14, 14),
        prompt_channel=1,
        pad_len=128,
    ) -> None:
        
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, 
                appearance_guidance=appearance_guidance_proj_dim, nheads=nheads, pooling_size=pooling_size, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.head = nn.Conv2d(text_guidance_proj_dim, 1, kernel_size=3, stride=1, padding=1)
        self.pad_len = pad_len

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def correlation(self, img_feats, text_feats, justtest = None):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # T P C
        if justtest is not None:
            justtest = F.normalize(justtest, dim=-1)
            corr2 = torch.einsum('bchw, tc -> bthw', img_feats, justtest)
        corr = torch.einsum('bchw, tpc -> bpthw', img_feats, text_feats)
        return corr

    def forward(self, img_feats, text_feats, justtest=None):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (T, P, C)
        """
        B, C, H, W = img_feats.shape
        if text_feats.ndim == 2:
            text_feats = text_feats.unsqueeze(dim=1)
        classes = None
        corr = self.correlation(img_feats, text_feats, justtest)
        if self.pad_len > 0 and text_feats.size(0) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, th_text.size(-2), th_text.size(-1)))
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            text_feats = th_text
            corr = torch.einsum('bchw, tpc -> bpthw', img_feats, th_text)

        corr_embed = self.corr_embed(corr) # B C T H W
        text_feats = text_feats.mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        projected_text_guidance = self.text_guidance_projection(text_feats)
        projected_guidance = self.guidance_projection(img_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)
        logits = self.head(corr_embed)
        return logits
    

if __name__ == "__main__":
    aggregator = Aggregator().to("cuda")
    img_feats = torch.randn(1, 512, 14, 14).cuda() # B C H W
    text_feats = torch.randn(16, 1, 512).cuda() # B T P C
    logits = aggregator(img_feats, text_feats) # B T H W