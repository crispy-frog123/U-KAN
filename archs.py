import torch
from torch import nn
import torch.nn.functional as F
import math

# ------------------------------------------------------
# 导入 timm 依赖 (处理版本兼容性)
# ------------------------------------------------------
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    except ImportError:
        from torch.nn import Identity as DropPath


        def to_2tuple(x):
            if isinstance(x, (list, tuple)):
                return x
            return (x, x)


        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            torch.nn.init.trunc_normal_(tensor, mean, std, a, b)
            return tensor

# 导入 KAN 和 FCSA 模块
from kan import KANLinear
from FCS_attention import MultiSpectralAttentionLayer

__all__ = ['UKAN']


class SpatialAttention(nn.Module):
    """
    论文 III.B.2) Spatial Attention
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FcsAttention(nn.Module):
    """
    论文 III.B. Frequency-Channel-Spatial Attention (FCSA)
    """

    def __init__(self, in_channels, out_channels, img_size, reduction=16):
        super(FcsAttention, self).__init__()

        # [修改] 增强鲁棒性，防止因通道数改变导致KeyError
        # 尝试根据通道数推断 DCT 尺寸，逻辑参考原代码：通道越小，分辨率越高
        # 原逻辑: out_channels//4 -> size//4, out_channels -> size//16

        c2wh = dict([
            (out_channels // 4, img_size // 4),
            (out_channels // 2, img_size // 8),
            (out_channels, img_size // 16),
            (out_channels * 2, img_size // 32)  # 补充可能的映射
        ])

        if out_channels in c2wh:
            dct_h = dct_w = c2wh[out_channels]
        else:
            # 兜底策略：如果找不到映射，默认使用 img_size // 16 或根据实际需求计算
            # 这是一个安全回退，防止崩溃
            dct_h = dct_w = img_size // 16

        self.spatial = SpatialAttention()

        # 论文 Eq (9) - (13)
        self.frequency_channel = MultiSpectralAttentionLayer(
            channel=out_channels,
            dct_h=dct_h,
            dct_w=dct_w,
            reduction=reduction,
            freq_sel_method='top16'
        )

    def forward(self, x):
        x = self.frequency_channel(x)
        x = self.spatial(x) * x
        return x


class ChannelLinear(nn.Module):
    """
    [优化] 使用 1x1 卷积替代 Linear，功能相同但无需 permute，效率更高
    用于 Multiscale Fusion 中的通道调整 (论文 Eq 14)
    """

    def __init__(self, in_channels, out_channels):
        super(ChannelLinear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ECAAttention(nn.Module):
    """Efficient Channel Attention (lightweight channel reweighting)."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class DW_bn_relu(nn.Module):
    """
    深度可分离卷积模块
    论文 Eq (5): LN -> KAN -> DwConv
    """

    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class KANLayer(nn.Module):
    """
    KAN层：实现论文中的 Tokenized KAN 逻辑
    对应论文 Eq (6): KAN(Z) = Phi_3(Phi_2(Phi_1(Z)))
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # KAN 参数
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            # 3层 KAN 结构
            self.fc1 = KANLinear(in_features, hidden_features, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                 base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc2 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                 base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc3 = KANLinear(hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                 scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                 base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # 对应每个 KAN 层后的处理
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # 3级串联结构
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x


class KANBlock(nn.Module):
    """
    Tok-KAN Block
    论文 Eq (5): Tok-KAN(X) = LN(Z) + DwConv(KAN(Z))
    注意：这里的实现实际上是 ResNet 风格的: x + DropPath(Layer(Norm(x)))
    """

    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """
    论文 III.A. Tokenization & Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class D_ConvLayer(nn.Module):
    """
    Decoder Convolution Layer
    """

    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UKAN(nn.Module):
    """
    [修改版] 3-Stage U-KAN: 适配小尺寸图像 (如 64x64)
    结构: 64->32->16->8 (Bottleneck) ->16->32->64
    """

    def __init__(self, num_classes, input_channels, deep_supervision=False,
                 img_size=64, patch_size=16,  # 注意 img_size 默认值
                 embed_dims=[128, 160, 256], no_kan=False,  # embed_dims 只需要第1个值起作用
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], **kwargs):  # depths 长度适配 3 stage
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.use_eca = kwargs.get('use_eca', False)
        eca_kernel = kwargs.get('eca_kernel', 3)

        # 基础通道数 (Base Channel)
        # 对应 Level 0: embed_dims[0] // 8
        # 对应 Level 1: embed_dims[0] // 4
        # 对应 Level 2: embed_dims[0]     <-- Bottleneck (Stage 3)
        base_dim = embed_dims[0]

        # --- 归一化层 (保留前3层) ---
        self.norm0 = norm_layer(base_dim // 8)
        self.norm1 = norm_layer(base_dim // 4)
        self.norm2 = norm_layer(base_dim)  # Bottleneck norm

        # Decoder 归一化 (保留对应层)
        self.dnorm2 = norm_layer(base_dim // 4)  # p2 output norm
        self.dnorm1 = norm_layer(base_dim // 8)  # p1 output norm

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # --- KAN Blocks (Encoder) ---
        # Stage 1
        self.block01 = nn.ModuleList(
            [KANBlock(dim=base_dim // 8, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])
        # Stage 2
        self.block12 = nn.ModuleList(
            [KANBlock(dim=base_dim // 4, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])
        # Stage 3 (Bottleneck)
        self.block23 = nn.ModuleList([KANBlock(dim=base_dim, drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer)])

        # --- KAN Blocks (Decoder) ---
        # Decode Stage 2
        self.dblock23 = nn.ModuleList(
            [KANBlock(dim=base_dim // 4, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])
        # Decode Stage 1
        self.dblock12 = nn.ModuleList(
            [KANBlock(dim=base_dim // 8, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        # --- Patch Embed (Encoder Downsampling) ---
        # Stage 1: Input -> Level 0
        self.patch_embed0 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=input_channels,
                                       embed_dim=base_dim // 8)
        # Stage 2: Level 0 -> Level 1
        self.patch_embed1 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=base_dim // 8,
                                       embed_dim=base_dim // 4)
        # Stage 3: Level 1 -> Level 2 (Bottleneck)
        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=base_dim // 4,
                                       embed_dim=base_dim)

        # --- Decoder Convs (用于通道变换和特征提取) ---
        # Decode 3->2
        self.decoder3 = D_ConvLayer(base_dim, base_dim // 4)
        # Decode 2->1
        self.decoder4 = D_ConvLayer(base_dim // 4, base_dim // 8)
        # Final Expand
        self.decoder5 = D_ConvLayer(base_dim // 8, base_dim // 8)

        # --- Upsample (1x1 Conv 用于 Multiscale Fusion 的通道对齐) ---
        # Fusion 3->2
        self.upsample3 = nn.Conv2d(base_dim, base_dim // 4, 1)
        # Fusion 2->1
        self.upsample4 = nn.Conv2d(base_dim // 4, base_dim // 8, 1)

        # --- FCSA Modules ---
        # Skip Connections (Decoder Path)
        self.FCSA2s = FcsAttention(base_dim // 2, base_dim // 2, img_size)  # Input is cat(64, 64) = 128
        self.FCSA1s = FcsAttention(base_dim // 4, base_dim // 4, img_size)  # Input is cat(32, 32) = 64

        # Fusion Path
        self.FCSA2 = FcsAttention(base_dim // 2, base_dim // 2, img_size)
        self.FCSA1 = FcsAttention(base_dim // 4, base_dim // 4, img_size)

        # --- Channel Linear (1x1 Conv) ---
        # Decoder Path
        self.backdim2s = ChannelLinear(base_dim // 2, base_dim // 4)
        self.backdim1s = ChannelLinear(base_dim // 4, base_dim // 8)

        # Fusion Path
        self.backdim2 = ChannelLinear(base_dim // 2, base_dim // 4)
        self.backdim1 = ChannelLinear(base_dim // 4, base_dim // 8)

        if self.use_eca:
            self.eca2s = ECAAttention(base_dim // 4, k_size=eca_kernel)
            self.eca1s = ECAAttention(base_dim // 8, k_size=eca_kernel)
            self.eca2 = ECAAttention(base_dim // 4, k_size=eca_kernel)
            self.eca1 = ECAAttention(base_dim // 8, k_size=eca_kernel)

        # --- Final Head ---
        self.final = nn.Conv2d(base_dim // 8, num_classes, kernel_size=1)

        # --- Deep Supervision Heads ---
        if self.deep_supervision:
            # 只有 p2 和 p1 两个中间输出
            self.head_p2 = nn.Conv2d(base_dim // 4, num_classes, kernel_size=1)
            self.head_p1 = nn.Conv2d(base_dim // 8, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        # ================= Encoder Path (3 Stages) =================
        # Stage 1
        out, H, W = self.patch_embed0(x)
        for i, blk in enumerate(self.block01): out = blk(out, H, W)
        out = self.norm0(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t1 = out  # Level 0 feature

        # Stage 2
        out, H, W = self.patch_embed1(out)
        for i, blk in enumerate(self.block12): out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out  # Level 1 feature

        # Stage 3 (Bottleneck)
        out, H, W = self.patch_embed2(out)
        for i, blk in enumerate(self.block23): out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p3 = out  # Level 2 feature (Bottleneck)

        # ================= Decoder Path =================
        # Decode Stage 2 (Using p3)
        # Conv p3 -> Upsample -> Cat t2
        out = F.relu(F.interpolate(self.decoder3(p3), scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t2), dim=1)  # Cat
        out = self.FCSA2s(out)  # Attention
        out = self.backdim2s(out)  # Channel reduction
        if self.use_eca:
            out = self.eca2s(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock23): out = blk(out, H, W)  # KAN Block

        # p2 Output
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p2 = out

        # Decode Stage 1 (Using p2)
        out = F.relu(F.interpolate(self.decoder4(p2), scale_factor=(2, 2), mode='bilinear'))
        out = torch.cat((out, t1), dim=1)  # Cat
        out = self.FCSA1s(out)
        out = self.backdim1s(out)
        if self.use_eca:
            out = self.eca1s(out)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock12): out = blk(out, H, W)

        # p1 Output
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p1 = out

        # ================= Multiscale Feature Fusion Path (Bottom-up) =================

        # Fusion Stage 2 (p3 upsampled + p2)
        # 修复点：这里明确使用 p3 作为最高层输入的起点，因为原先的 Stage 4/5 已经移除
        out = self.upsample3(p3)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p2), dim=1)
        out = self.FCSA2(out)
        out = self.backdim2(out)
        if self.use_eca:
            out = self.eca2(out)
        fusion2 = out  # 保存中间结果，方便下一级使用

        # Fusion Stage 1 (fusion2 upsampled + p1)
        out = self.upsample4(fusion2)  # 使用上一级融合的结果
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p1), dim=1)
        out = self.FCSA1(out)
        out = self.backdim1(out)
        if self.use_eca:
            out = self.eca1(out)
        fusion1 = out

        # Final Convolution
        out = F.relu(F.interpolate(self.decoder5(fusion1), scale_factor=(2, 2), mode='bilinear'))
        final_out = self.final(out)

        if self.deep_supervision:
            input_size = x.shape[2:]
            # Deep supervision heads for p2 and p1
            out_p2 = F.interpolate(self.head_p2(p2), size=input_size, mode='bilinear', align_corners=False)
            out_p1 = F.interpolate(self.head_p1(p1), size=input_size, mode='bilinear', align_corners=False)

            return [out_p2, out_p1, final_out]  # 权重建议: [0.4, 0.4, 1.0]

        return final_out
