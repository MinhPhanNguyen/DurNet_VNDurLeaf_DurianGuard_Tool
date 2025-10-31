import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F
# ===========================
# DurNet: MobileNetV3-Small + Tiny ViT + Dropout
# ===========================
class DurNet(nn.Module):
    def __init__(
        self, 
        num_classes=6, 
        img_size=224, 
        patch_dim=128, 
        num_transformer_layers=2, 
        dropout_rate=0.5
    ):
        super().__init__()
        # ------------------------
        # MobileNetV3-Small backbone
        # ------------------------
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone_features = self.backbone.features  # [B, C, H', W']
        self.backbone_out_channels = 576  # output channel của MobileNetV3-Small

        # ------------------------
        # Patch embedding
        # ------------------------
        self.patch_proj = nn.Linear(self.backbone_out_channels, patch_dim)
        self.patch_dropout = nn.Dropout(dropout_rate)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, patch_dim))  # tạm thời, resize trong forward

        # ------------------------
        # Tiny Transformer
        # ------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_dim,
            nhead=4,
            dim_feedforward=patch_dim * 2,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # ------------------------
        # Classification head
        # ------------------------
        self.norm = nn.LayerNorm(patch_dim)
        self.head_dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(patch_dim, num_classes)

        # ------------------------
        # Weight initialization
        # ------------------------
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # ------------------------
        # Backbone feature extraction
        # ------------------------
        f = self.backbone_features(x)  # [B, C, H, W]
        B, C, H, W = f.shape

        # Flatten H*W thành tokens
        tokens = f.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.patch_proj(tokens)       # [B, N, D]
        tokens = self.patch_dropout(tokens)

        # Thêm CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)  # [B, N+1, D]

        # Resize positional embedding nếu cần
        if self.pos_embed.shape[1] != x.shape[1]:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, x.shape[1], self.pos_embed.shape[2], device=x.device)
            )
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classification using CLS token
        cls_out = self.norm(x[:, 0])
        cls_out = self.head_dropout(cls_out)
        out = self.head(cls_out)
        return out