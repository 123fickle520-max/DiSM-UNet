import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.unet_stvit import UNet_STA
from unet.unet_dinov3 import UNet_DINOv3


class ChannelAttention(nn.Module):
    """通道注意力模块 (SE-Block)
    动态学习每个通道的重要性，用于 MFEblock 输出后的特征过滤
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class GlobalGAM(nn.Module):
    """语义引导的全局仿射调制 (Semantic-Guided Global GAM)
    """
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, 1, bias=True)  # 输出 gamma 和 beta
        )
        
        # 零初始化：确保初始阶段 gamma=0, beta=0，即 S_tilde = S
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, d_align, S):
        # d_align: 空间对齐的 DINOv3 语义特征
        # S: 投影后的 STViT 医学纹理特征
        
        global_semantic = self.gap(d_align)  # [B, C, 1, 1]
        affine_params = self.mlp(global_semantic)  # [B, 2C, 1, 1]
        
        gamma, beta = torch.chunk(affine_params, 2, dim=1)  # 各自 [B, C, 1, 1]
        
        # 仿射调制
        S_tilde = S * (1 + gamma) + beta
        return S_tilde


class MFEBlock(nn.Module):
    """多特征提取模块 (Multi-Feature Extraction Block)
    并行多尺度感受野提取，针对腹部器官尺度差异极大的问题
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 分支 1: 细粒度通道平滑
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 分支 2: 标准解剖边界捕获
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 分支 3: 大范围上下文依赖 (空洞卷积)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 融合后的通道注意力
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # 多尺度特征相加融合
        out = out1 + out2 + out3
        
        # 滤除冗余
        out = self.ca(out)
        return out


class GMFE_Fusion(nn.Module):
    """G-MFE: Global-GAM & Multi-Feature Extraction 融合模块
    """
    def __init__(self, stvit_dim, dinov3_dim, out_dim, scale_level):
        super().__init__()
        self.scale_level = scale_level

        # STViT特征投影
        self.stvit_proj = nn.Conv2d(stvit_dim, out_dim, 1, bias=False)

        # DINOv3特征的级联反卷积对齐
        if scale_level == 1:
            self.dinov3_upsample = nn.Sequential(
                nn.ConvTranspose2d(dinov3_dim, dinov3_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dinov3_dim // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(dinov3_dim // 2, dinov3_dim // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dinov3_dim // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(dinov3_dim // 4, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
        elif scale_level == 2:
            self.dinov3_upsample = nn.Sequential(
                nn.ConvTranspose2d(dinov3_dim, dinov3_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(dinov3_dim // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(dinov3_dim // 2, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
        elif scale_level == 3:
            self.dinov3_upsample = nn.Sequential(
                nn.ConvTranspose2d(dinov3_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.dinov3_upsample = nn.Sequential(
                nn.Conv2d(dinov3_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )

        # 阶段1: 全局语义仿射调制
        self.global_gam = GlobalGAM(out_dim)
        
        # 阶段2: 多特征提取
        self.mfe_block = MFEBlock(in_channels=out_dim * 2, out_channels=out_dim)
        
        # 阶段3: 残差聚合输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, stvit_feat, dinov3_feat):
        # 1. 投影与对齐
        S = self.stvit_proj(stvit_feat)
        d_align = self.dinov3_upsample(dinov3_feat)

        if d_align.shape[2:] != S.shape[2:]:
            d_align = F.interpolate(d_align, size=S.shape[2:], mode='bilinear', align_corners=False)

        # 2. 全局仿射调制 (Global-GAM)
        S_tilde = self.global_gam(d_align, S)

        # 3. 异构拼接与多特征提取 (MFE)
        cat_feat = torch.cat([S_tilde, d_align], dim=1)
        F_mfe = self.mfe_block(cat_feat)

        # 4. 残差聚合 (抗遗忘机制，保护底层解剖边界)
        out = self.final_conv(F_mfe + S_tilde)

        return out


class DualEncoderV3(nn.Module):
    """双编码器V3 - G-MFE 融合架构
    """
    def __init__(self, n_in=1, n_class=9, img_size=224):
        super().__init__()

        # ==================== STViT编码器 ====================
        stvit_full = UNet_STA(n_in=n_in, n_class=n_class)
        self.stvit_e1 = stvit_full.e1
        self.stvit_svl1 = stvit_full.e_svl1
        self.stvit_e2 = stvit_full.e2
        self.stvit_svl2 = stvit_full.e_svl2
        self.stvit_e3 = stvit_full.e3
        self.stvit_svl3 = stvit_full.e_svl3
        self.stvit_e4 = stvit_full.e4
        self.stvit_svl4 = stvit_full.e_svl4

        # ==================== DINOv3编码器 ====================
        dinov3_model = UNet_DINOv3(n_in=n_in, n_class=n_class, img_size=img_size, vit_model='vitl16')
        self.dinov3_encoder = dinov3_model.encoder

        # ==================== G-MFE 融合模块 ====================
        self.fusion1 = GMFE_Fusion(stvit_dim=64, dinov3_dim=1024, out_dim=64, scale_level=1)
        self.fusion2 = GMFE_Fusion(stvit_dim=128, dinov3_dim=1024, out_dim=128, scale_level=2)
        self.fusion3 = GMFE_Fusion(stvit_dim=256, dinov3_dim=1024, out_dim=256, scale_level=3)
        self.fusion4 = GMFE_Fusion(stvit_dim=512, dinov3_dim=1024, out_dim=512, scale_level=4)

        # ====================  G-MFE 特征级联传递与 Fusion 模块 ====================
        # 建立 G-MFE-1 到 4 的逐层传递，使得 f1-f3 信息层层下卷并融入 Bottleneck
        self.trans1_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.trans2_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.trans3_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # 对应模型图中的粉色 Fusion 模块
        self.fusion_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # ==================== 解码器 ====================
        self.bottleneck = stvit_full.b
        self.d1 = stvit_full.d1
        self.d_svl1 = stvit_full.d_svl1
        self.d2 = stvit_full.d2
        self.d_svl2 = stvit_full.d_svl2
        self.d3 = stvit_full.d3
        self.d_svl3 = stvit_full.d_svl3
        self.d4 = stvit_full.d4
        self.d_svl4 = stvit_full.d_svl4
        self.outputs = stvit_full.outputs

    def forward(self, x):
        # --- STViT 路径 ---
        s1, p1 = self.stvit_e1(x)
        stv1 = self.stvit_svl1(p1)
        s2, p2 = self.stvit_e2(stv1)
        stv2 = self.stvit_svl2(p2)
        s3, p3 = self.stvit_e3(stv2)
        stv3 = self.stvit_svl3(p3)
        s4, p4 = self.stvit_e4(stv3)
        stv4 = self.stvit_svl4(p4)

        # --- DINOv3 路径 ---
        dinov3_features = self.dinov3_encoder(x)

        # --- G-MFE 跨流融合 ---
        f1 = self.fusion1(stv1, dinov3_features[0])
        f2 = self.fusion2(stv2, dinov3_features[1])
        f3 = self.fusion3(stv3, dinov3_features[2])
        f4 = self.fusion4(stv4, dinov3_features[3])

        # --- G-MFE 特征级联传递与 Fusion ---
        # 对应模型图底部的黑色连线：f_i 逐层下采样并与 f_i+1 累加传递
        f1_to_2 = self.trans1_2(f1)
        f2_fused = f2 + f1_to_2

        f2_to_3 = self.trans2_3(f2_fused)
        f3_fused = f3 + f2_to_3

        f3_to_4 = self.trans3_4(f3_fused)
        f4_fused = f4 + f3_to_4

        # 经过图示中红粉色的 Fusion Block 模块
        f_final = self.fusion_block(f4_fused)

        # --- 解码路径 ---
        # 融合了所有 G-MFE 模块传递信息的 f_final 进入 bottleneck
        b = self.bottleneck(f_final)
        
        d1 = self.d1(b, s4)
        d1 = self.d_svl1(d1)
        d2 = self.d2(d1, s3)
        d2 = self.d_svl2(d2)
        d3 = self.d3(d2, s2)
        d3 = self.d_svl3(d3)
        d4 = self.d4(d3, s1)
        d4 = self.d_svl4(d4)

        out = self.outputs(d4)
        return out

    def load_pretrained_weights(self, stvit_path, dinov3_path):
        """加载预训练权重（复用V2兼容逻辑）"""
        print("\n" + "="*80)
        print("Loading pretrained weights for Dual-Encoder V3 (G-MFE) model...")
        print("="*80)

        # 加载STViT部分
        stvit_ckpt = torch.load(stvit_path, map_location='cpu')
        stvit_state = stvit_ckpt.get('model_state_dict', stvit_ckpt)
        
        stvit_encoder_state = {}
        bottleneck_decoder_state = {}
        
        for k, v in stvit_state.items():
            if k.startswith(('e1', 'e2', 'e3', 'e4', 'e_svl')):
                new_k = k.replace('e_svl', 'stvit_svl').replace('e1', 'stvit_e1').replace('e2', 'stvit_e2').replace('e3', 'stvit_e3').replace('e4', 'stvit_e4')
                stvit_encoder_state[new_k] = v
            elif k.startswith(('b.', 'd1', 'd2', 'd3', 'd4', 'd_svl', 'outputs')):
                if k.startswith('b.'):
                    bottleneck_decoder_state['bottleneck.' + k[2:]] = v
                else:
                    bottleneck_decoder_state[k] = v

        self.load_state_dict(stvit_encoder_state, strict=False)
        self.load_state_dict(bottleneck_decoder_state, strict=False)

        # 加载DINOv3部分
        dinov3_ckpt = torch.load(dinov3_path, map_location='cpu')
        dinov3_state = dinov3_ckpt.get('model', dinov3_ckpt.get('state_dict', dinov3_ckpt))
        dinov3_mapped = {}
        for k, v in dinov3_state.items():
            new_k = None
            if k.startswith('backbone.'):
                new_k = k.replace('backbone.', 'dinov3_encoder.')
            elif k.startswith('blocks.') or k.startswith('patch_embed.') or k.startswith('norm.'):
                new_k = 'dinov3_encoder.' + k
            elif k.startswith('encoder.'):
                new_k = 'dinov3_' + k

            if new_k and new_k in self.state_dict():
                if v.shape == self.state_dict()[new_k].shape:
                    dinov3_mapped[new_k] = v

        self.load_state_dict(dinov3_mapped, strict=False)
        print("Pretrained weights loaded successfully!")


def create_dual_encoder_v3(n_in=1, n_class=9, img_size=224, stvit_path=None, dinov3_path=None):
    model = DualEncoderV3(n_in, n_class, img_size)
    if stvit_path and dinov3_path:
        model.load_pretrained_weights(stvit_path, dinov3_path)
    return model


if __name__ == '__main__':
    print("Testing Dual-Encoder V3 (G-MFE) model with cascade transmission...")
    model = create_dual_encoder_v3(n_in=1, n_class=9, img_size=224)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {out.shape}")
    print("\n✓ G-MFE Cascade Transmission Model structure test passed!")