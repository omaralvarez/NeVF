# 🍦 Vanilla PyTorch
import torch
from torch import nn

# 📊 Data
import numpy as np
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed_b, pe_embed_l):
        super().__init__()

        if pe_embed_b == 0:
            self.embed_length = 1
            self.pe_embed = False
        else:
            self.lbase = torch.tensor(pe_embed_b)
            self.levels = torch.tensor(pe_embed_l)
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels
            self.pe_embed = True

            b = self.lbase ** torch.arange(self.levels) * np.pi
            self.register_buffer("b", b, persistent=False)

    def forward(self, x):
        if self.pe_embed is False:
            return x[:, None]
        else:
            # Multiply positions by embedding encodings
            xb = x.unsqueeze(1) * self.b
            # Apply sin and cos with extra dimensions
            enc = torch.stack([torch.sin(xb), torch.cos(xb)], dim=2)

            # Interleave tensors using extra dimension: [[sin, cos, sin, cos...],...]
            return enc.view(-1, self.embed_length)


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError(
                "__init__() missing 1 required positional argument: 'upscale_factor'"
            )

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f"pixel_shuffle expects input to have at least 3 dimensions, but got {x.ndim} dimension(s)"
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f"pixel_shuffle expects its input's 'channel' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}"
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor**3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(30 * input)  # see SIREN paper for the factor 30


def ActivationLayer(act_type):
    if act_type == "relu":
        act_layer = nn.ReLU()
    elif act_type == "leaky":
        act_layer = nn.LeakyReLU()
    elif act_type == "leaky01":
        act_layer = nn.LeakyReLU(negative_slope=0.1)
    elif act_type == "relu6":
        act_layer = nn.ReLU6()
    elif act_type == "gelu":
        act_layer = nn.GELU()
    elif act_type == "sin":
        act_layer = Sin()
    elif act_type == "swish":
        act_layer = nn.SiLU()
    elif act_type == "softplus":
        act_layer = nn.Softplus()
    elif act_type == "hardswish":
        act_layer = nn.Hardswish()
    elif act_type is None:
        act_layer = nn.Identity()
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):
    if norm_type is None:
        norm_layer = nn.Identity()
    elif norm_type == "bn":
        norm_layer = nn.BatchNorm3d(num_features=ch_width)
    elif norm_type == "in":
        norm_layer = nn.InstanceNorm3d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Memory efficient attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(
                dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            )
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x

        return x


class NeVF_MLP(nn.Module):
    def __init__(self, dim_list, act="relu", bias=True):
        super().__init__()

        act_fn = ActivationLayer(act)

        fc_list = []
        for i in range(len(dim_list) - 1):
            fc_list += [nn.Linear(dim_list[i], dim_list[i + 1], bias=bias), act_fn]

        self.model = nn.Sequential(*fc_list)

    def forward(self, x):
        return self.model(x)


class NeVF_CustomConv(nn.Module):
    def __init__(self, **kwargs):
        super(NeVF_CustomConv, self).__init__()

        ngf, new_ngf, stride = kwargs["ngf"], kwargs["new_ngf"], kwargs["stride"]
        self.conv_type = kwargs["conv_type"]
        if self.conv_type == "conv":
            self.conv = nn.Conv3d(
                ngf, new_ngf * stride * stride * stride, 3, 1, 1, bias=kwargs["bias"]
            )
            self.up_scale = PixelShuffle3d(stride)
        elif self.conv_type == "deconv":
            self.conv = nn.ConvTranspose3d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == "bilinear":
            self.conv = nn.Upsample(
                scale_factor=stride, mode="bilinear", align_corners=True
            )
            self.up_scale = nn.Conv3d(
                ngf, new_ngf, 2 * stride + 1, 1, stride, bias=kwargs["bias"]
            )

    def forward(self, x):
        out = self.conv(x)

        return self.up_scale(out)


class NeVFBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv = NeVF_CustomConv(
            ngf=kwargs["ngf"],
            new_ngf=kwargs["new_ngf"],
            stride=kwargs["stride"],
            bias=kwargs["bias"],
            conv_type=kwargs["conv_type"],
        )
        self.norm = NormLayer(kwargs["norm"], kwargs["new_ngf"])
        self.act = ActivationLayer(kwargs["act"])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv_Up_Block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        ngf = kwargs["ngf"]
        new_ngf = kwargs["new_ngf"]

        if ngf <= new_ngf:
            factor = 8

            self.conv1 = NeVF_CustomConv(
                ngf=ngf,
                new_ngf=ngf // factor,
                stride=kwargs["stride"],
                bias=kwargs["bias"],
                conv_type=kwargs["conv_type"],
            )
            self.conv2 = nn.Conv3d(ngf // factor, new_ngf, 3, 1, 1, bias=kwargs["bias"])
        else:
            self.conv1 = nn.Conv3d(ngf, new_ngf, 3, 1, 1, bias=kwargs["bias"])
            self.conv2 = NeVF_CustomConv(
                ngf=new_ngf,
                new_ngf=new_ngf,
                stride=kwargs["stride"],
                bias=kwargs["bias"],
                conv_type=kwargs["conv_type"],
            )
        self.norm = NormLayer(kwargs["norm"], kwargs["new_ngf"])
        self.act = ActivationLayer(kwargs["act"])

    def forward(self, x):
        return self.act(self.norm(self.conv2(self.conv1(x))))
