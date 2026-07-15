from .base import BaseModel
from .modules import (
    PositionalEncoding,
    TransformerBlock,
    NeVF_MLP,
    NeVFBlock,
    Conv_Up_Block,
)

# 🍦 Vanilla PyTorch
import torch
import torch.nn as nn

# 🏋️‍♀️ Weihts & Biases
import wandb


class NeVF_Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        # t mapping
        self.pe_t = PositionalEncoding(
            pe_embed_b=wandb.config["pos_b"], pe_embed_l=wandb.config["pos_l"]
        )

        stem_dim_list = [int(x) for x in wandb.config["stem_dim_num"].split("_")]
        self.fc_d, self.fc_w, self.fc_h, self.fc_dim = [
            int(x) for x in wandb.config["fc_hw_dim"].split("_")
        ]
        self.block_dim = wandb.config["block_dim"]

        mlp_dim_list = [self.pe_t.embed_length] + stem_dim_list + [self.block_dim]
        self.stem_t = NeVF_MLP(dim_list=mlp_dim_list, act=wandb.config["act"])

        # xy mapping
        xy_coord = torch.stack(
            torch.meshgrid(
                torch.arange(self.fc_d) / self.fc_d,
                torch.arange(self.fc_h) / self.fc_h,
                torch.arange(self.fc_w) / self.fc_w,
                indexing="ij",
            ),
            dim=0,
        ).flatten(
            1, 3
        )  # [3, d*h*w]

        self.pe_xy = PositionalEncoding(
            pe_embed_b=wandb.config["xypos_b"], pe_embed_l=wandb.config["xypos_l"]
        )

        x_coord = self.pe_xy(xy_coord[0])
        y_coord = self.pe_xy(xy_coord[1])
        z_coord = self.pe_xy(xy_coord[2])
        self.register_buffer(
            "xy_emb_cached", torch.cat([x_coord, y_coord, z_coord], dim=1)
        )

        self.stem_xy = NeVF_MLP(
            dim_list=[3 * self.pe_xy.embed_length, self.block_dim],
            act=wandb.config["act"],
        )

        self.trans1 = TransformerBlock(
            dim=self.block_dim,
            heads=1,
            dim_head=64,
            mlp_dim=wandb.config["mlp_dim"],
            dropout=0.0,
            prenorm=False,
        )

        self.trans2 = TransformerBlock(
            dim=self.block_dim,
            heads=8,
            dim_head=64,
            mlp_dim=wandb.config["mlp_dim"],
            dropout=0.0,
            prenorm=False,
        )

        if self.block_dim == self.fc_dim:
            self.toconv = nn.Identity()
        else:
            self.toconv = NeVF_MLP(
                dim_list=[self.block_dim, self.fc_dim], act=wandb.config["act"]
            )

        # Build conv layers
        self.layers, self.head_layers, self.t_layers, self.norm_layers = [
            nn.ModuleList() for _ in range(4)
        ]

        ngf = self.fc_dim
        for i, stride in enumerate(wandb.config["stride_list"]):
            if i == 0:
                # Expand channel width at first stage
                new_ngf = int(ngf * wandb.config["expansion"])
            else:
                # Change the channel width for each stage
                new_ngf = max(
                    ngf // (1 if stride == 1 else wandb.config["reduction"]),
                    wandb.config["lower_width"],
                )

            self.t_layers.append(
                NeVF_MLP(dim_list=[128, 2 * ngf], act=wandb.config["act"])
            )

            self.norm_layers.append(nn.InstanceNorm3d(ngf, affine=False))

            if i == 0:
                self.layers.append(
                    Conv_Up_Block(
                        ngf=ngf,
                        new_ngf=new_ngf,
                        stride=stride,
                        bias=wandb.config["bias"],
                        norm=wandb.config["norm"],
                        act=wandb.config["act"],
                        conv_type=wandb.config["conv_type"],
                    )
                )
            else:
                self.layers.append(
                    NeVFBlock(
                        ngf=ngf,
                        new_ngf=new_ngf,
                        stride=stride,
                        bias=wandb.config["bias"],
                        norm=wandb.config["norm"],
                        act=wandb.config["act"],
                        conv_type=wandb.config["conv_type"],
                    )
                )
            ngf = new_ngf

            # Build head classifier, upscale feature layer, upscale img layer
            head_layer = [None]
            if wandb.config["sin_res"]:
                if i == len(wandb.config["stride_list"]) - 1:
                    head_layer = nn.Conv3d(
                        ngf, self.channels, 1, 1, bias=wandb.config["bias"]
                    )
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv3d(
                    ngf, self.channels, 1, 1, bias=wandb.config["bias"]
                )
            self.head_layers.append(head_layer)

        self.sigmoid = wandb.config["sigmoid"]

        self.T_num = 20
        self.pe_t_manipulate = PositionalEncoding(
            pe_embed_b=wandb.config["pos_b_tm"], pe_embed_l=wandb.config["pos_l_tm"]
        )
        self.t_branch = NeVF_MLP(
            dim_list=[self.pe_t_manipulate.embed_length, 128, 128],
            act=wandb.config["act"],
        )

    def fuse_t(self, x, t):
        # x: [B, C, D, H, W], normalized among C
        # t: [B, 3*C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]
        beta = t[:, f_dim:]

        gamma = gamma[..., None, None, None]
        beta = beta[..., None, None, None]
        out = torch.addcmul(beta, x, gamma)
        return out

    def forward_impl(self, input_id):
        t = input_id

        t_emb = self.stem_t(self.pe_t(t))  # [B, L]
        t_manipulate = self.t_branch(self.pe_t_manipulate(t))

        xy_emb = self.xy_emb_cached

        # Apply stem and first transformer to unbatched sequence [1, seq, L] to save compute
        xy_emb = self.stem_xy(xy_emb).unsqueeze(0)
        xy_emb = self.trans1(xy_emb)

        # Expand to batch size
        xy_emb = xy_emb.expand(t_emb.shape[0], -1, -1)  # [B, h*w, L]

        # Fuse t into xy map efficiently via broadcasting
        emb = xy_emb * t_emb.unsqueeze(1)
        emb = self.toconv(self.trans2(emb))

        emb = emb.reshape(emb.shape[0], self.fc_d, self.fc_h, self.fc_w, emb.shape[-1])
        emb = emb.permute(0, 4, 1, 2, 3)
        output = emb

        out_list = []
        for i, (layer, head_layer, t_layer, norm_layer) in enumerate(
            zip(self.layers, self.head_layers, self.t_layers, self.norm_layers)
        ):
            # t_manipulate
            output = norm_layer(output)
            t_feat = t_layer(t_manipulate)
            output = self.fuse_t(output, t_feat)
            # conv
            output = layer(output)
            if head_layer is not None and i == len(self.layers) - 1:
                img_out = head_layer(output)
                # Normalize the final output iwth sigmoid or tanh function
                img_out = (
                    torch.sigmoid(img_out)
                    if self.sigmoid
                    else (torch.tanh(img_out) + 1) * 0.5
                )
                out_list.append(img_out)

        return out_list

    def forward(self, coords, ts):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_list = self.forward_impl(ts)  # a list containing [B or 2B, 3, H, W]
        return output_list[-1]


class NeVFModel(BaseModel):
    def __init__(self):
        super().__init__()
        if self.single_decoder:
            self.model = NeVF_Generator(wandb.config["features_out"])
        else:
            self.model_p = NeVF_Generator(wandb.config["depth"])
            self.model_u = NeVF_Generator(wandb.config["depth"])
            self.model_v = NeVF_Generator(wandb.config["depth"])
            self.model_w = NeVF_Generator(wandb.config["depth"])
