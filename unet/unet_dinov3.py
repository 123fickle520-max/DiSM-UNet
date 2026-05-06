"""
DINOv3-STA-UNet: Fusion of DINOv3 ViT encoder with STA-UNet decoder
Author: Auto-generated for research purposes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
import math


class DINOv3Encoder(nn.Module):
    """
    DINOv3 ViT-based encoder that extracts multi-scale features
    Compatible with STA-UNet decoder architecture
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Patch embedding - modified for grayscale input
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Feature extraction indices (extract from blocks at 1/4, 2/4, 3/4, 4/4 of depth)
        # For vits16 (depth=12): blocks 3, 6, 9, 12 -> indices [2, 5, 8, 11]
        # For vitl16 (depth=24): blocks 6, 12, 18, 24 -> indices [5, 11, 17, 23]
        interval = depth // 4
        self.feature_indices = [interval - 1, interval * 2 - 1, interval * 3 - 1, depth - 1]

        # Grid size (for 224x224 with patch_size=16: 14x14)
        self.grid_size = img_size // patch_size

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Extract features from intermediate blocks
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.feature_indices:
                # Remove CLS token and reshape to spatial format
                feat = x[:, 1:, :]  # (B, num_patches, embed_dim)
                feat = feat.permute(0, 2, 1).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
                features.append(feat)

        # features: list of 4 tensors, each (B, embed_dim, 14, 14) for 224x224 input
        return features


class FeatureProjection(nn.Module):
    """
    Project and upsample DINOv3 features to match STA-UNet decoder expectations
    """
    def __init__(self, in_dim, out_dim, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        # Projection layer
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # Upsampling if needed
        if scale_factor > 1:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_dim, out_dim, kernel_size=scale_factor*2,
                                 stride=scale_factor, padding=scale_factor//2, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.upsample(x)
        return x


class conv_block(nn.Module):
    """Basic conv block from original STA-UNet"""
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """Up-convolution block from original STA-UNet"""
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """Attention block from original STA-UNet (simplified version)"""
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class UNet_DINOv3(nn.Module):
    """
    DINOv3-STA-UNet: Uses DINOv3 ViT as encoder, keeps STA-UNet decoder

    Args:
        n_in: Number of input channels (1 for grayscale medical images)
        n_class: Number of output classes
        img_size: Input image size (default: 224)
        vit_model: 'vits16' or 'vitl16' for different DINOv3 variants
    """
    def __init__(self, n_in=1, n_class=9, img_size=224, vit_model='vits16'):
        super(UNet_DINOv3, self).__init__()

        # DINOv3 configuration
        if vit_model == 'vits16':
            embed_dim = 384
            depth = 12
            num_heads = 6
        elif vit_model == 'vitl16':
            embed_dim = 1024
            depth = 24
            num_heads = 16
        else:
            raise ValueError(f"Unknown vit_model: {vit_model}")

        # DINOv3 Encoder
        self.encoder = DINOv3Encoder(
            img_size=img_size, patch_size=16, in_chans=n_in,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )

        # Feature projection layers to match STA-UNet decoder expectations
        # DINOv3 outputs features at 14x14 (stride 16 for 224x224 input)
        # We need: s1(64 @ 112x112), s2(128 @ 56x56), s3(256 @ 28x28), s4(512 @ 14x14)

        self.proj4 = FeatureProjection(embed_dim, 512, scale_factor=1)   # 14x14 -> 14x14
        self.proj3 = FeatureProjection(embed_dim, 256, scale_factor=2)   # 14x14 -> 28x28
        self.proj2 = FeatureProjection(embed_dim, 128, scale_factor=4)   # 14x14 -> 56x56
        self.proj1 = FeatureProjection(embed_dim, 64, scale_factor=8)    # 14x14 -> 112x112

        # Bottleneck: downsample s4 further to get 1024 channels at 7x7
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(512, 1024)
        )

        # Decoder blocks (from original STA-UNet)
        self.Up4 = up_conv(1024, 512)
        self.Att4 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv4 = conv_block(1024, 512)

        self.Up3 = up_conv(512, 256)
        self.Att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv3 = conv_block(512, 256)

        self.Up2 = up_conv(256, 128)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv2 = conv_block(256, 128)

        self.Up1 = up_conv(128, 64)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv1 = conv_block(128, 64)

        self.Conv = nn.Conv2d(64, n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # DINOv3 Encoder: extract multi-scale features
        features = self.encoder(x)  # List of 4 features from blocks [3, 6, 9, 12]

        # Project features to match decoder expectations
        s1 = self.proj1(features[0])   # (B, 64, 112, 112)
        s2 = self.proj2(features[1])   # (B, 128, 56, 56)
        s3 = self.proj3(features[2])   # (B, 256, 28, 28)
        s4 = self.proj4(features[3])   # (B, 512, 14, 14)

        # Bottleneck
        d5 = self.bottleneck(s4)       # (B, 1024, 7, 7)

        # Decoder with attention and skip connections
        d4 = self.Up4(d5)              # (B, 512, 14, 14)
        s4 = self.Att4(g=d4, x=s4)
        d4 = torch.cat((s4, d4), dim=1)
        d4 = self.Up_conv4(d4)         # (B, 512, 14, 14)

        d3 = self.Up3(d4)              # (B, 256, 28, 28)
        s3 = self.Att3(g=d3, x=s3)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.Up_conv3(d3)         # (B, 256, 28, 28)

        d2 = self.Up2(d3)              # (B, 128, 56, 56)
        s2 = self.Att2(g=d2, x=s2)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.Up_conv2(d2)         # (B, 128, 56, 56)

        d1 = self.Up1(d2)              # (B, 64, 112, 112)
        s1 = self.Att1(g=d1, x=s1)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.Up_conv1(d1)         # (B, 64, 112, 112)

        out = self.Conv(d1)            # (B, n_class, 112, 112)

        # Upsample to original size
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out

    def load_dinov3_pretrained(self, pretrained_path, strict=False):
        """
        Load DINOv3 pretrained weights with adaptation for grayscale input

        Args:
            pretrained_path: Path to DINOv3 pretrained weights (.pth file)
            strict: Whether to strictly enforce key matching
        """
        print(f"Loading DINOv3 pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Get current model state dict for shape comparison
        model_state_dict = self.encoder.state_dict()

        # Filter and adapt encoder weights
        encoder_state_dict = {}
        skipped_keys = []

        for k, v in state_dict.items():
            # Remove any prefix like 'module.' or 'model.'
            k = k.replace('module.', '').replace('model.', '')

            # Map to our encoder
            if k.startswith('patch_embed') or k.startswith('blocks.') or \
               k.startswith('norm') or k in ['cls_token', 'pos_embed']:

                # Special handling for patch embedding: adapt RGB (3-channel) to grayscale (1-channel)
                if k == 'patch_embed.proj.weight' and k in model_state_dict:
                    # v shape: [embed_dim, 3, patch_size, patch_size]
                    # model expects: [embed_dim, 1, patch_size, patch_size]
                    if v.shape[1] == 3 and model_state_dict[k].shape[1] == 1:
                        # Average across RGB channels to get grayscale
                        adapted_weight = v.mean(dim=1, keepdim=True)
                        encoder_state_dict[k] = adapted_weight
                        print(f"  Adapted patch_embed.proj.weight: {v.shape} -> {adapted_weight.shape}")
                    elif v.shape == model_state_dict[k].shape:
                        encoder_state_dict[k] = v
                    else:
                        skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_state_dict[k].shape})")

                # For all other weights, check shape compatibility
                elif k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        encoder_state_dict[k] = v
                    else:
                        skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_state_dict[k].shape})")
                else:
                    # Key exists in checkpoint but not in model (e.g., extra layers)
                    skipped_keys.append(f"{k} (not in model)")

        # Load weights (non-strict to allow partial loading)
        msg = self.encoder.load_state_dict(encoder_state_dict, strict=False)

        # Print loading summary
        print(f"  Loaded {len(encoder_state_dict)} weights")
        if skipped_keys:
            print(f"  Skipped {len(skipped_keys)} incompatible weights")
            if len(skipped_keys) <= 5:
                for sk in skipped_keys:
                    print(f"    - {sk}")

        print(f"  Load result: {msg}")

        return msg


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing UNet_DINOv3 model...")
    model = UNet_DINOv3(n_in=1, n_class=9, img_size=224, vit_model='vits16').to(device)

    # Dummy input
    x = torch.randn(2, 1, 224, 224).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (2, 9, 224, 224)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nModel test passed!")
