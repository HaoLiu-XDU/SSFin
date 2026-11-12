# This file contains code adapted from third-party projects:
#   - MultiSpectralDCTLayer / MultiSpectralAttention (DCT frequency channel attention):
#     adapted from FcaNet (Qin et al., ICCV 2021), https://github.com/cfzd/FcaNet
#   - sparsemax: implementation of Martins & Astudillo, ICML 2016.
#   - spectral_spatial_block / block_1D: adapted from [Spectral-Spatial Mamba for HSI classification
#     ().
# Modifications were made by the authors of this repository.

from mamba import SSM
from functools import partial
from typing import Callable
from timm.models.layers import DropPath
from pos_embed import get_2d_sincos_pos_embed
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.spa = nn.Conv2d(2, 1, 1, padding=0, padding_mode='reflect', bias=True)


    def forward(self, x):

        x_avg = torch.mean(x, dim=1, keepdim=True)


        x_max, _ = torch.max(x, dim=1, keepdim=True)

        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.spa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        x_spa_c = x[:, :,1, 1]

        x_gap = x_gap + x_spa_c.unsqueeze(2).unsqueeze(3)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 1, padding=0, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(y)

        sattn = self.sa(x)

        pattn1 = sattn + cattn

        pattn2 = self.pa(initial, pattn1)

        result = initial + pattn2 * x + (1 - pattn2) * y

        result = self.conv(result)
        return result


def sparsemax(input, dim=-1):

    input = input.transpose(dim, -1)
    original_size = input.size()
    input = input.reshape(-1, input.size(-1))

    input_safe = torch.nan_to_num(input, nan=0.0, posinf=1e4, neginf=-1e4)


    zs, _ = torch.sort(input_safe, descending=True, dim=1)

    range_idx = torch.arange(1, input.size(1) + 1, device=input.device, dtype=input.dtype).view(1, -1)

    zs_cumsum = zs.cumsum(dim=1)

    support = (1 + range_idx * zs) > zs_cumsum

    k = support.sum(dim=1, keepdim=True).long()

    k = torch.clamp(k, min=1)

    tau = (zs_cumsum.gather(1, k - 1) - 1) / k.to(input.dtype)

    output = torch.clamp(input - tau, min=0)

    output = output.reshape(original_size)
    output = output.transpose(dim, -1)
    return output

class HFEST(nn.Module):
    def __init__(
            self,
            dim,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.position = nn.Parameter(torch.tensor(0.5))
        self.amplification = nn.Parameter(torch.tensor(0.05))
        self.proj = nn.Linear(576, 576)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(576)

    def forward(self, x_in):
        x = x_in
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        attn = (q @ k.transpose(-2, -1))
        attn1 = self.relu(attn) ** 2
        out = (attn1 @ v)
        attn_weights = sparsemax(out, dim=3)
        spectral_attn = (attn_weights * out).sum(dim=3, keepdim=True)  # (b, 1, 64, 1)
        spectral_attn = spectral_attn.squeeze(3).squeeze(1)
        out = rearrange(out, 'b head c (h w) -> b (head c h w) ', head=self.num_heads, h=h, w=w)
        out = self.norm(out)
        b = x_in.shape[0]
        x_in_reshaped = x_in.reshape(b, 64, 9)
        fft_result = torch.fft.fft(x_in_reshaped, dim=1)
        gain = self.amplification
        freq_idx = torch.linspace(0.0, 1.0, 64).to(x_in.device).view(1, C, 1)  # shape (1,C,1)
        d = torch.abs(freq_idx - 0.5)  # in [0, 0.5]
        pos_raw = self.position
        pos = pos_raw /2
        kappa = 100.0
        soft_mask = torch.sigmoid(kappa * (pos - d))
        high_pass_filtered = fft_result * soft_mask
        spectral_attn = spectral_attn.unsqueeze(-1)
        spectral_attn_fre = torch.fft.fft(spectral_attn, dim=1)
        amplified_fft = high_pass_filtered * gain * (1 + spectral_attn_fre)
        processed_signal = torch.fft.ifft(amplified_fft, dim=1).real
        x_fre = processed_signal + x_in_reshaped
        x_fre = x_fre.reshape(b, -1)
        out2 = self.norm(x_fre)
        combined = (out+out2)/10
        out = self.proj(combined)+x_in.reshape(-1,576)
        return out

def p_attention(A, B, temperature=0.01):
    orr = A * B
    corr = orr / torch.sqrt(torch.tensor(A.size(2)).to(B))
    attention = torch.softmax(corr / temperature, dim=-1)
    return attention

def fft_with_phase_amplitude(input_images,eps=1e-8):

    input_images = input_images.permute(0, 3, 1, 2).float()
    fft_images = torch.fft.fft2(input_images, dim=(-2,-1))
    re, im = fft_images.real, fft_images.imag
    mag2 = re * re + im * im
    # amplitude: eps inside the sqrt keeps the gradient finite at 0
    amplitude = torch.sqrt(mag2 + eps)
    valid = mag2 > eps
    re = torch.where(valid, re, torch.ones_like(re))
    im = torch.where(valid, im, torch.zeros_like(im))
    phase = torch.where(valid, torch.atan2(im, re), torch.zeros_like(re))


    return amplitude, phase


def ifft_with_reconstruction(amplitude, phase):

    fft_complex = amplitude * torch.cos(phase) + 1j * amplitude * torch.sin(phase)
    ifft_images = torch.fft.ifft2(fft_complex, dim=(-2, -1))

    return ifft_images.real.permute(0, 2, 3, 1)


class block_1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        bi: bool = True,
        cls: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.bi = bi
        self.cls = cls

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x1 = self.self_attention(x)

        if self.bi:
            if self.cls:
                x2 = x[:,0:-1,:]
                cls_token = x[:,-1:,:]
                x2 = torch.flip(x2, dims=[1])
                x2 = torch.cat((x2, cls_token), dim=1)
                x3 = self.self_attention(x2)

                x2 = x3[:,0:-1,:]
                cls_token = x3[:,-1:,:]
                x3 = torch.flip(x2, dims=[1])
                x3 = torch.cat((x3, cls_token), dim=1)

            else:
                x3 = torch.flip(x, dims=[1])
                x3 = self.self_attention(x3)
                x3 = torch.flip(x3, dims=[1])
            return self.drop_path((x1+x3)/2) + input
        else:
            return self.drop_path(x1) + input

def positional_embedding_1d(seq_len, embed_size):
    position_enc = torch.zeros(seq_len, embed_size)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
    position_enc[:, 0::2] = torch.sin(position.float() * div_term)
    position_enc[:, 1::2] = torch.cos(position.float() * div_term)
    return position_enc.unsqueeze(0)

class spectral_spatial_block(nn.Module):
    def __init__(self, embed_dim, bi=False, N=8, drop_path=0.0, norm_layer=nn.LayerNorm, cls = True, fu = True):
        super(spectral_spatial_block, self).__init__()
        self.spa_block = block_1D(
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls,

            )
        self.spe_block = block_1D(
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls
            )
        self.linear = nn.Linear(N, N)
        self.norm = norm_layer(embed_dim)
        self.l1= nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias = False),
            nn.Sigmoid(),)
        self.fu = fu
    def forward(self, x_spa, x_spe):
        x_spa2 = self.spa_block(x_spa)
        B, N, D = x_spa.shape
        x_spe2 = self.spe_block(x_spe)
        _,N1,_ = x_spe.shape
        if self.fu:
            x_spa_c = x_spa2[:,(N-1)//2,:]
            x_spe_c = x_spe2.mean(1)
            sig = self.l1((x_spa_c+x_spe_c)/2).unsqueeze(1)

            x_spa = x_spa*sig.expand(-1,N,-1)
            x_spe = x_spe*sig.expand(-1,N1,-1)
        return x_spa, x_spe


def select_top_bottom_from_canonical_map_to_hw(
    target_h: int,
    target_w: int,
    base_grid: int = 9,
    top_percent: float = 0.25,
    bottom_percent: float = 0.25,
    include_dc_in_low: bool = True,
) -> Tuple[List[int], List[int], List[int], List[int]]:

    coords = [(u, v) for u in range(base_grid) for v in range(base_grid)]
    # compute radial magnitude
    freq_mag = [(math.sqrt(u*u + v*v), u, v) for (u, v) in coords]
    # sort ascending for low, descending for high
    freq_sorted_desc = sorted(freq_mag, key=lambda x: x[0], reverse=True)
    freq_sorted_asc = sorted(freq_mag, key=lambda x: x[0], reverse=False)

    total = base_grid * base_grid
    n_top = max(1, int(round(total * top_percent)))
    n_bot = max(1, int(round(total * bottom_percent)))

    top_uv = [(u, v) for (_, u, v) in freq_sorted_desc[:n_top]]
    bot_uv = [(u, v) for (_, u, v) in freq_sorted_asc[:n_bot]]

    # ensure DC in low if required
    if include_dc_in_low and (0, 0) not in bot_uv:
        if len(bot_uv) >= n_bot:
            bot_uv[-1] = (0, 0)
        else:
            bot_uv.append((0, 0))

    # mapping function
    def map_and_dedup(uv_list):
        mapped = []
        seen = set()
        for (u, v) in uv_list:
            u_m = int(round(u * target_h / base_grid))
            v_m = int(round(v * target_w / base_grid))
            u_m = max(0, min(target_h - 1, u_m))
            v_m = max(0, min(target_w - 1, v_m))
            if (u_m, v_m) not in seen:
                mapped.append((u_m, v_m))
                seen.add((u_m, v_m))
        if len(mapped) == 0:
            # fallback to DC
            mapped = [(0, 0)]
        mx, my = zip(*mapped)
        return list(mx), list(my)

    mapper_x_high, mapper_y_high = map_and_dedup(top_uv)
    mapper_x_low, mapper_y_low = map_and_dedup(bot_uv)
    return mapper_x_high, mapper_y_high, mapper_x_low, mapper_y_low


class MultiSpectralDCTLayer(nn.Module):
    """
    Build DCT basis filters for a list of (u,v) frequencies mapped to (H,W),
    then expand each basis into a contiguous block of channels. Channel allocation
    is done evenly with remainder assigned to the first groups.
    Returns (B, C) after spatial projection.
    """
    def __init__(self, H: int, W: int, mapper_x: List[int], mapper_y: List[int], channel: int):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        self.H = H
        self.W = W
        self.mapper_x = mapper_x
        self.mapper_y = mapper_y
        self.num_freq = len(mapper_x)
        self.channel = channel
        base = channel // self.num_freq
        rem = channel % self.num_freq
        channel_allocation = [base + (1 if i < rem else 0) for i in range(self.num_freq)]
        self.channel_allocation = channel_allocation  # keep for debug
        weight = torch.zeros(channel, H, W, dtype=torch.float32)
        ch_idx = 0
        for i, (u, v) in enumerate(zip(mapper_x, mapper_y)):
            n_ch = channel_allocation[i]
            # build DCT 2D basis
            phi_x = torch.tensor([self.build_filter(t, u, H) for t in range(H)], dtype=torch.float32)
            phi_y = torch.tensor([self.build_filter(t, v, W) for t in range(W)], dtype=torch.float32)
            basis = torch.outer(phi_x, phi_y)  # shape (H, W)
            if n_ch > 0:
                weight[ch_idx:ch_idx + n_ch, :, :] = basis.unsqueeze(0).expand(n_ch, -1, -1)
                ch_idx += n_ch
        # register as buffer
        self.register_buffer('weight', weight)  # (C, H, W)

    @staticmethod
    def build_filter(pos, freq, POS):
        """
        same formula as FCANet / DCT:
        cos(pi * freq * (pos + 0.5) / POS) / sqrt(POS), scaled by sqrt(2) if freq > 0
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        # x: (B, C, H, W)
        assert x.dim() == 4
        B, C, H, W = x.shape
        # expect H==self.H, W==self.W or caller applied pooling
        # element-wise multiply and spatial sum
        # self.weight: (C, H, W) -> will broadcast on batch dim
        xw = x * self.weight  # (B, C, H, W)
        res = torch.sum(xw, dim=(2, 3))  # (B, C)
        return res

class MultiSpectralAttentionPercent(nn.Module):
    """
    Build an attention module that uses selected frequency set (mapped to HxW),
    then passes projection through channel MLP to generate (B, C, 1, 1) attention.
    This is a single-mode layer; create one instance for high and one for low.
    """
    def __init__(self,
                 channel: int,
                 dct_h: int,
                 dct_w: int,
                 mapper_x: List[int],
                 mapper_y: List[int],
                 reduction: int = 8):
        super().__init__()
        self.channel = channel
        self.dct_h = dct_h
        self.dct_w = dct_w
        # instantiate DCT layer
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        mid = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_pooled = x
        if H != self.dct_h or W != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        # DCT projection -> (B, C)
        y = self.dct_layer(x_pooled)
        # channel MLP
        y = self.fc(y).view(B, C, 1, 1)
        out = x * (y + 1)
        return out, y
class AFCM(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, embed_dim=64, norm_layer=None, flatten = True):
        super().__init__()
        self.embed_dim = 64
        self.grid_size = (3, 3)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1))
        self.proj_back = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=3))
        self.norm = nn.Identity()
        mxh, myh, mxl, myl = select_top_bottom_from_canonical_map_to_hw(
            target_h=9, target_w=9, base_grid=9, top_percent=0.25, bottom_percent=0.25, include_dc_in_low=True
        )
        self.att_high = MultiSpectralAttentionPercent(embed_dim, 9, 9, mxh, myh, reduction=8)
        self.att_low = MultiSpectralAttentionPercent(embed_dim, 9, 9, mxl, myl, reduction=8)
        self.mapper_high = (mxh, myh)
        self.mapper_low = (mxl, myl)
    def forward(self, x):

        B = x.shape[0]
        x_proj = self.proj(x)
        out_high, y_high = self.att_high(x_proj)
        out_high = self.proj_back(out_high)
        tokens = out_high.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        _, y_low = self.att_low(x_proj)
        return tokens, y_low

class SSFIN(nn.Module):
    def __init__(self, spa_img_size=(224, 224),spe_img_size=(5,5), spa_patch_size=16, spe_patch_size=2, in_chans=3, hid_chans = 32, embed_dim=128, nclass=10, xband=4, drop_path=0.0, depth=4, bi=True,
                 norm_layer=nn.LayerNorm, global_pool=False, cls = True, fu=True):
        super().__init__()
        self.convP = nn.Sequential(nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=1))
        self.averagePool = nn.AvgPool2d(kernel_size=3)
        self.maxPooling = nn.MaxPool2d(kernel_size=3)
        self.conv2d = nn.Conv2d(embed_dim, embed_dim, kernel_size=(2, 1))
        self.layer = nn.Sequential(nn.Linear(embed_dim, embed_dim//2),
                               nn.Linear(embed_dim//2, embed_dim),
                                   nn.ReLU(inplace=True))
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            )
        self.lf= nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.Sigmoid(), )
        self.spe_AFCM = AFCM(spa_img_size, spa_patch_size, xband, embed_dim)
        self.blocksX = nn.ModuleList([
            block_1D(hidden_dim=embed_dim, drop_path=drop_path, bi=bi, cls=cls) for i in range(depth)
        ])
        self.normX = norm_layer(embed_dim)
        self.half_spa_patch_size = spa_img_size[0] // 2
        self.half_spe_patch_size = spe_img_size[0] // 2
        self.spa_AFCM = AFCM(spa_img_size, spa_patch_size, hid_chans, embed_dim)
        spa_num_patches = self.spa_AFCM.num_patches
        if in_chans % spe_patch_size ==0:
          spe_num_patches = in_chans//spe_patch_size
        else:
          spe_num_patches = in_chans//spe_patch_size

        self.cls = cls
        if self.cls:
          self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, 64))
          self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          N = spa_num_patches+spe_num_patches+2
          self.cs = -1
        else:
          N = spa_num_patches+spe_num_patches
          self.cs = N

        self.blocks = nn.ModuleList([
                spectral_spatial_block(embed_dim, bi, N=N, drop_path = drop_path, cls = self.cls, fu = fu) for i in range(depth)
                        ])
        self.head = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim),
                                  nn.Linear(embed_dim, nclass))
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, spa_num_patches+1, embed_dim), requires_grad=False)
        self.spe_pos_embed = nn.Parameter(positional_embedding_1d(10, 64), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, spa_num_patches + 1, embed_dim), requires_grad=False)
        self.HFEST = HFEST(64, 1)
        self.fusion = CGAFusion(64)
        self.fc_norm1 = norm_layer(embed_dim)
        self.fc_norm2 = norm_layer(embed_dim)
        self.fc_norm3 = norm_layer(embed_dim)
        self.fc_norm4 = norm_layer(embed_dim)
        self.fc_norm5 = norm_layer(embed_dim)
        self.fc_norm6 = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        spa_pos_embed = get_2d_sincos_pos_embed(self.spa_pos_embed.shape[-1], int(self.spa_AFCM.num_patches**.5), cls_token=True)
        self.spa_pos_embed.data.copy_(torch.from_numpy(spa_pos_embed).float().unsqueeze(0))

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.spe_AFCM.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def lidar_branch(self, x):
        x = x + self.pos_embed[:, :-1, :]
        if self.cls:
            cls_token = self.cls_token + self.pos_embed[:, -1:, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)
        for blk in self.blocksX:
            x = blk(x)
        return x


    def hsi_branch(self, x_AFCM,x_redu):
        '''embed_dim=64, after conv, spatial size is 3*3'''
        x_HFEST = self.HFEST(x_redu[:, :,
                                     self.half_spa_patch_size - self.half_spe_patch_size:self.half_spa_patch_size + self.half_spe_patch_size + 1,
                                     self.half_spa_patch_size - self.half_spe_patch_size:self.half_spa_patch_size + self.half_spe_patch_size + 1])
        x_HFEST_r = x_HFEST.reshape(-1, 64, 3, 3)
        x_AFCM_r = x_AFCM.reshape(-1, 3, 3, 64).permute(0, 3, 1, 2)
        output_spe = self.fusion(x_AFCM_r, x_HFEST_r)
        x_spe = output_spe.permute(0, 2, 3, 1).reshape(-1, 9, 64)

        # x_AFCM_reshape, x_HFEST_reshape = x_AFCM.reshape(-1, 3, 3, 64).permute(0, 3, 2, 1), x_HFEST.reshape(-1, 3, 3, 64).permute(0,3,2,1)
        # output_spe = self.fusion(x_AFCM_reshape, x_HFEST_reshape)
        # x_spe = output_spe.permute(0, 3, 2, 1).reshape(output_spe.shape[0], 9, 64)


        x_spe = x_spe + self.spe_pos_embed[:, :-1, :]
        x_spa = x_AFCM + self.spa_pos_embed[:, :-1, :]
        spa_cls_token = self.spa_cls_token + self.spa_pos_embed[:, -1:, :]
        spa_cls_tokens = spa_cls_token.expand(x_spa.shape[0], -1, -1)
        spe_cls_token = self.spe_cls_token + self.spe_pos_embed[:, -1:, :]
        spe_cls_tokens = spe_cls_token.expand(x_spe.shape[0], -1, -1)
        x_spe = torch.cat((x_spe, spe_cls_tokens), dim=1)
        x_spa = torch.cat((x_spa, spa_cls_tokens), dim=1)
        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)
        x_spa_fusion, x_spe_fusion = x_spa, x_spe
        return x_spa_fusion, x_spe_fusion

    def HFRM(self,x_spa3,lidar3):
        A, P = fft_with_phase_amplitude(x_spa3)
        Ax, Px = fft_with_phase_amplitude(lidar3)
        '''embed_dim=64, after conv, spatial size is 3*3'''
        similar = p_attention(P, Px, temperature=0.01)
        attention = similar.reshape(-1, 64, 9)
        P_fusion = self.convP(torch.angle(torch.exp(1j * P) + torch.exp(1j * Px)))
        A_fusion = (A + Ax) / 2
        plus = attention * (A_fusion.reshape(-1, 64, 9))
        A_fusion = A_fusion + 0.2 * plus.reshape(-1, 64, 3, 3)
        A_fusion1 = self.averagePool(A_fusion)
        A_fusion2 = self.maxPooling(A_fusion)
        A_weight = torch.cat((A_fusion1, A_fusion2), dim=2)
        A_weight = self.conv2d(A_weight)
        A_weight = self.layer(A_weight.squeeze(-1).squeeze(-1))
        A_weight = nn.Softmax(dim=1)(A_weight)
        A_weight = A_weight.reshape(-1, 64, 1, 1)
        A_fusion = (1 + A_weight) * ((A + Ax) / 2)
        x_spa2 = ifft_with_reconstruction(A_fusion, P_fusion)
        fus = x_spa2.permute(0, 3, 1, 2).reshape(-1, 64, 9)
        sig2 = self.lf(fus.mean(dim=2)).unsqueeze(2)
        fu = fus.shape[1]
        fus = fus * (sig2.expand(-1, fu, -1))
        x_spa2 = self.fc_norm3(fus.mean(dim=2))
        return x_spa2

    def forward(self, x, lidar,return_branch=False):
        x_spa_redu = self.dimen_redu(x)
        # AFCM for multisource fusion and enhance
        x_spa, weight_low_hsi = self.spa_AFCM(x_spa_redu)
        lidar, weight_low_lidar = self.spe_AFCM(lidar)
        weight_fusion_low = (weight_low_hsi+weight_low_lidar)/2
        weight_fusion_low = weight_fusion_low.squeeze(2).squeeze(2).unsqueeze(1)
        x_spa_AFCM = x_spa * (1+weight_fusion_low.expand_as(x_spa))
        lidar_AFCM = lidar * weight_fusion_low.expand_as(lidar)
        #hsi branch-SSAF
        x_spa_fusion, x_spe_fusion = self.hsi_branch(x_spa_AFCM, x_spa_redu)
        # sar/lidar branch
        lidar_cls = self.lidar_branch(lidar_AFCM)
        # remove class token
        x_spa_fusion, x_spe_fusion = x_spa_fusion[:, 0:self.cs, :], x_spe_fusion[:, 0:self.cs, :]
        lidar = lidar_cls[:, 0:self.cs, :]
        # fuse the two sources
        x_SSAF = self.fc_norm1((x_spa_fusion.mean(dim=1) + x_spe_fusion.mean(dim=1))/2)
        # lidar branch's output
        lidar2 = lidar.mean(dim=1)
        # High-frequency resonance mask
        x_spa3=x_spa_fusion.reshape(-1,3,3,64)
        lidar3=lidar.reshape(-1,3,3,64)
        x_spa2 = self.HFRM(x_spa3,lidar3)
        # cat all the features
        x = torch.cat((x_SSAF, x_spa2, lidar2), dim=1)
        out = self.head(x)
        return out




