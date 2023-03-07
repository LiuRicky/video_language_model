import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple), shape of qkv is (B, #head, N, C//#head)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(768, 96)
        self.activate = nn.Sigmoid()
        self.hidden_proj = nn.Linear(96, 96)
        self.up_proj = nn.Linear(96, 768)
    
    def forward(self, x):
        x = self.down_proj(x)
        x = self.activate(x)
        x = self.hidden_proj(x)
        x = self.up_proj(x)
        return x

class AdapterBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.pre_adapter = Adapter()
        self.mid_adapter = Adapter()

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.pre_adapter(x)
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.mid_adapter(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class STAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(768, 384)
        self.adapter = nn.Conv3d(384,384,(3,3,3),padding=1)
        self.up_proj = nn.Linear(384, 768)
    
    def forward(self, x):
        x = self.down_proj(x)
        B_origin, N_origin, C = x.shape # # shape here is (#videos*N, frames, C)
        frames = 8
        x = x.reshape(B_origin//frames, frames, N_origin, C) # shape here is (#videos, frames, N_origin, C)
        H = int((N_origin-1)**0.5)
        x_cls = x[:,:,:1,:]
        x_space = x[:,:,1:,:]
        x_space = x_space.reshape(x.shape[0], x.shape[1], H, H, x.shape[3]).permute(0,4,1,2,3) # shape here is (#videos, C, frames, H, W)
        # print('shape 132: ', x_space.shape)
        x_space = self.adapter(x_space)
        # print('shape 134: ', x_space.shape)
        x_space = x_space.permute(0,2,3,4,1).reshape(x.shape[0], x.shape[1], -1, x.shape[3]) # shape here is (#videos, frames, N_origin-1, C)
        x = torch.cat([x_cls, x_space],dim=2).reshape(B_origin, N_origin, C)
        x = self.up_proj(x)
        return x

class STAdapterBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.st_adapter = STAdapter()

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.st_adapter(x)
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class FusionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
        # self.st_scale = nn.Parameter(torch.zeros([1,1,1,1]))
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        # uniform sorted
        B_origin, N_origin, C = x.shape # # shape here is (#videos*N, frames, C)
        frames = 8
        x = x.reshape(B_origin//frames, frames, N_origin, C).reshape(B_origin//frames, -1, C) # shape here is (#videos, frames*N, C)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # shape of qkv is (#videos, #head, frames*N, C//#head)

        indices_cls_frame = torch.arange(0, N, N_origin) # all [CLS] tokens
        indices_per_frame = [torch.arange(i*N_origin+1+i, (i+1)*N_origin, frames) for i in range(frames)]
        indices_selected = torch.cat([indices_cls_frame] + indices_per_frame, dim=0)
        # indices_selected = torch.cat([indices_per_frame + i*N_origin for i in range(frames)], dim=0)
        k = k[:,:,indices_selected,:]
        v = v[:,:,indices_selected,:]
        # print(k.shape)

        # st_time = time.time()
        attn = (q @ k.transpose(-2, -1)) * self.scale # shape of attn is (#videos, #head, frames*N, #selected)
        #### we can add a mask here, surpress token from other frames  ####
        token_mask = torch.zeros(attn.shape).to(attn.device)
        cls_len = indices_cls_frame.shape[0]
        token_mask[:,:,:,:cls_len] = 1 # set [CLS] as 1
        st = cls_len
        ed = cls_len + indices_per_frame[0].shape[0]
        token_mask[:,:,0:N_origin,st:ed] = 1
        for i in range(1, len(indices_per_frame)):
            st = st + indices_per_frame[i-1].shape[0]
            ed = ed + indices_per_frame[i].shape[0]
            token_mask[:,:,i*N_origin:(i+1)*N_origin,st:ed] = 1
        surpress_mask = (1-token_mask)*0.8
        token_mask = token_mask + surpress_mask
        attn = attn * token_mask
        #### we can add a mask here, surpress token from other frames  ####


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # ed_time = time.time()
        # print('attn time: ', ed_time - st_time)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(B, frames, N_origin, C).reshape(B_origin, N_origin, C)
        return x

class FusionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FusionAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.num_frm = 8
        # self.temp_attn_scale = nn.Parameter(torch.zeros([]))
        # self.temp_pool = TemporalPooling(pool_size=3)
        
        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

        # # x shape here is (bs*n_frames, n_patches, hid_dim)
        # x_norm = self.norm1(x)
        # x_spatial = self.drop_path(self.attn(x_norm, register_hook=register_hook))

        # x_temporal = self.permuate_spatial2temporal(x_norm, self.num_frm)
        # x_temporal = self.drop_path(self.temp_pool(x_temporal))
        # x_temporal = self.permuate_temporal2spatial(x_temporal, x.size(1))

        # x = x + x_spatial + F.tanh(self.temp_attn_scale) * x_temporal
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        # return x

    def permuate_spatial2temporal(self, x, num_frames):
        B, P, H = x.shape  # x shape here is (bs*n_frames, n_patches, hid_dim)
        x = x.reshape(-1, num_frames, P, H)  # shape here is (bs, n_frames, n_patches, hid_dim)
        x = x.permute(0,2,1,3).reshape(-1, num_frames, H)
        return x

    def permuate_temporal2spatial(self, x, num_tokens):
        B, T, H = x.shape  # x shape here is (bs*n_patches, n_frames, hid_dim)
        x = x.reshape(-1, num_tokens, T, H)  # shape here is (bs, n_patches, n_frames, hid_dim)
        x = x.permute(0,2,1,3).reshape(-1, num_tokens, H)
        return x

class SpatialPooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3, stride=2):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=stride)

    def forward(self, x):
        '''
        x: shape (B, L, D)
        '''
        x = x.permute(0,2,1)
        # x = self.pool(x)
        x_spatial = self.pool(x[:,:,1:])
        x_cls = x[:,:,:1]
        x = torch.cat([x_cls, x_spatial], dim=-1)
        x = x.permute(0,2,1)
        return x

class TemporalPooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        '''
        x: shape (B, L, D)
        '''
        x = x.permute(0,2,1)
        x = self.pool(x)
        x = x.permute(0,2,1)
        return x

class DualBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.num_frm = 8
        self.temp_norm1 = norm_layer(dim)
        self.temp_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temp_attn_scale = nn.Parameter(torch.zeros([]))
        # self.temp_pool = TemporalPooling(pool_size=3)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        # x shape here is (bs*n_frames, n_patches, hid_dim)
        x_norm = self.norm1(x)

        x_temporal = self.permuate_spatial2temporal(x_norm, self.num_frm)
        x_temporal = self.drop_path(self.temp_attn(x_temporal, register_hook=register_hook))
        x_temporal = self.permuate_temporal2spatial(x_temporal, x.size(1))

        x_spatial = self.drop_path(self.attn(x_norm, register_hook=register_hook))

        x = x + x_spatial + F.tanh(self.temp_attn_scale) * x_temporal
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 

    def permuate_spatial2temporal(self, x, num_frames):
        B, P, H = x.shape  # x shape here is (bs*n_frames, n_patches, hid_dim)
        x = x.reshape(-1, num_frames, P, H)  # shape here is (bs, n_frames, n_patches, hid_dim)
        x = x.permute(0,2,1,3).reshape(-1, num_frames, H)
        return x

    def permuate_temporal2spatial(self, x, num_tokens):
        B, T, H = x.shape  # x shape here is (bs*n_patches, n_frames, hid_dim)
        x = x.reshape(-1, num_tokens, T, H)  # shape here is (bs, n_patches, n_frames, hid_dim)
        x = x.permute(0,2,1,3).reshape(-1, num_tokens, H)
        return x

class SpatialFusionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.pooling = SpatialPooling(pool_size=2)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False):
        x = self.pooling(x)
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 use_grad_checkpointing=False, ckpt_layer=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ### for temporal pos_embed ####
        # self.temp_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # ### for temporal pos_embed ####

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer)
            )
            if i < 11 else #  (i != 5 and i != 11) else # (i != 3 and i != 6 and i != 9)
            FusionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer)
            ) 
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.compress_pooling = SpatialPooling(pool_size=3)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def shuffle_patches(self, x, num_frames):
        B_orign, P_orign, H = x.shape  # x shape here is (bs*n_frames, n_patches, hid_dim)
        x = x.view(-1, num_frames, x.size(1), x.size(2))  # shape here is (bs, n_frames, n_patches, hid_dim)

        # ######### shuffle cls ##############
        # x_space = x[:,:,1:,:] # shape here is (bs, n_frames, n_patches-1, hid_dim)
        # x_cls = x[:,:,0,:]  # shape here is (bs, n_frames, hid_dim)
        # B, F, H = x_cls.shape
        # x_cls = x_cls.permute(0,2,1)
        # x_cls = x_cls.reshape(B, F, H)
        # '''invert operation
        # x_cls.reshape(B, H, F).permute(0,2,1)
        # '''
        # x = torch.cat((x_cls.unsqueeze(2), x_space), dim=2)
        # x = x.view(B_orign, P_orign, H)
        # ######### shuffle cls ##############

        ######### shuffle tokens ##############
        x = x.permute(0,2,1,3).reshape(B_orign, P_orign, H)
        ######### shuffle tokens ##############
        
        return x

    def unshuffle_patches(self, x, num_frames):
        B_orign, P_orign, H = x.shape # x shape here is (bs*n_frames, n_patches, hid_dim)
        x = x.view(-1, num_frames, x.size(1), x.size(2))  # shape here is (bs, n_frames, n_patches, hid_dim)
        
        # ######### shuffle cls ##############
        # x_space = x[:,:,1:,:] # shape here is (bs, n_frames, n_patches-1, hid_dim)
        # x_cls = x[:,:,0,:]  # shape here is (bs, n_frames, hid_dim)
        # B, F, H = x_cls.shape
        # x_cls.reshape(B, H, F).permute(0,2,1)
        # x = torch.cat((x_cls.unsqueeze(2), x_space), dim=2)
        # x = x.view(B_orign, P_orign, H)
        # ######### shuffle cls ##############

        ######### shuffle tokens ##############
        B, F, P, H = x.shape
        x = x.reshape(B, P, F, H).permute(0,2,1,3).reshape(B_orign, P_orign, H)
        ######### shuffle tokens ##############

        return x

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)    # x shape here is (bs*n_frames, n_patches, hid_dim)

        # ### for temp pos embed ###
        # num_frames = 8
        # x = x.view(-1, num_frames, x.size(1), x.size(2)) # shape here is (bs, n_frames, n_patches, hid_dim)
        # x = x + self.temp_pos_embed[:, :num_frames, :].unsqueeze(2)
        # x = x.view(B, x.size(2), x.size(3))
        # ### for temp pos embed ###

        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i)
        x = self.norm(x)

        # x = self.compress_pooling(x)
        
        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint


### useless code ###
        # # #### make k, v append global pool ####
        # # print("k shape before: ", k.shape)
        # frames = 8
        # shapes = k.shape
        # k_global_avg = k.reshape(-1, frames, shapes[1], shapes[2], shapes[3]) # shape here is (#videos, frames, #head, N, C//#head)
        # k_global_avg = torch.mean(k_global_avg, 1, True).expand(-1, frames, -1, -1, -1) # shape here is  (#videos, 1*frames, #head, N, C//#head)
        # k_global_avg = k_global_avg.reshape(shapes) # shape here is (B, #head, N, C//#head)
        # k = torch.cat([k, k_global_avg], dim=2)
        # # print("k shape after: ", k.shape)

        # v_global_avg = v.reshape(-1, frames, shapes[1], shapes[2], shapes[3]) # shape here is (#videos, frames, #head, N, C//#head)
        # v_global_avg = torch.mean(v_global_avg, 1, True).expand(-1, frames, -1, -1, -1) # shape here is  (#videos, 1*frames, #head, N, C//#head)
        # v_global_avg = v_global_avg.reshape(shapes) # shape here is (B, #head, N, C//#head)
        # v = torch.cat([v, v_global_avg], dim=2)
        # # #### make k, v append global pool ####