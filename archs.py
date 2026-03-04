import torch
from torch import nn
import torch.nn.functional as F
import math

# ------------------------------------------------------
# Optional timm dependency with compatibility fallback.
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

# Import KAN and FCSA components.
from kan import KANLinear
from FCS_attention import MultiSpectralAttentionLayer

__all__ = ['UKAN']


class SpatialAttention(nn.Module):
    """Spatial attention module (Section III.B.2)."""

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
    """Frequency-Channel-Spatial Attention (FCSA) module."""

    def __init__(self, in_channels, out_channels, img_size, reduction=16):
        super(FcsAttention, self).__init__()
        # Infer DCT resolution from channel width with a robust fallback.

        c2wh = dict([
            (out_channels // 4, img_size // 4),
            (out_channels // 2, img_size // 8),
            (out_channels, img_size // 16),
            (out_channels * 2, img_size // 32)  # Extended mapping for deeper channels.
        ])

        if out_channels in c2wh:
            dct_h = dct_w = c2wh[out_channels]
        else:
            # Safe fallback when no predefined mapping is available.
            dct_h = dct_w = img_size // 16

        self.spatial = SpatialAttention()
        # Corresponds to Eqs. (9)-(13) in the manuscript.
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
    """Channel projection with 1x1 convolution for multiscale fusion (Eq. 14)."""

    def __init__(self, in_channels, out_channels):
        super(ChannelLinear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DW_bn_relu(nn.Module):
    """Depthwise convolution refinement block used after KAN transforms (Eq. 5)."""

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
    """Tokenized KAN layer implementing Eq. (6): KAN(Z) = Phi_3(Phi_2(Phi_1(Z)))."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # KAN hyperparameters.
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            # Three-layer KAN stack.
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

        # Post-KAN depthwise refinement blocks.
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
        # Three-stage serial transform.
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
    """Tok-KAN block (Eq. 5) implemented in residual form."""

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
    """Tokenization and patch embedding module (Section III.A)."""

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
    """Three-stage U-KAN architecture for compact inputs (e.g., 64x64)."""

    def __init__(self, num_classes, input_channels, deep_supervision=False,
                 img_size=64, patch_size=16,  # Default image size.
                 embed_dims=[128, 160, 256], no_kan=False,  # Width is primarily determined by embed_dims[0].
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], **kwargs):  # Depth definition for the three-stage model.
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Base channel schedule:
        # Level 0: embed_dims[0] // 8
        # Level 1: embed_dims[0] // 4
        # Level 2: embed_dims[0] (bottleneck)
        base_dim = embed_dims[0]

        # Encoder normalization layers.
        self.norm0 = norm_layer(base_dim // 8)
        self.norm1 = norm_layer(base_dim // 4)
        self.norm2 = norm_layer(base_dim)  # Bottleneck norm

        # Decoder normalization layers.
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
        # Decoder convolution blocks for channel adaptation.
        # Decode 3->2
        self.decoder3 = D_ConvLayer(base_dim, base_dim // 4)
        # Decode 2->1
        self.decoder4 = D_ConvLayer(base_dim // 4, base_dim // 8)
        # Final Expand
        self.decoder5 = D_ConvLayer(base_dim // 8, base_dim // 8)
        # 1x1 upsample adapters for multiscale fusion channel alignment.
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

        # --- Final Head ---
        self.final = nn.Conv2d(base_dim // 8, num_classes, kernel_size=1)

        # --- Deep Supervision Heads ---
        if self.deep_supervision:
            # Intermediate supervision heads for p2 and p1.
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
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock12): out = blk(out, H, W)

        # p1 Output
        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p1 = out

        # ================= Multiscale Feature Fusion Path (Bottom-up) =================

        # Fusion Stage 2 (p3 upsampled + p2)
        # Use p3 as the highest-level source feature after stage reduction.
        out = self.upsample3(p3)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p2), dim=1)
        out = self.FCSA2(out)
        out = self.backdim2(out)
        fusion2 = out  # Cache intermediate fusion result for the next stage.

        # Fusion Stage 1 (fusion2 upsampled + p1)
        out = self.upsample4(fusion2)  # Propagate fusion result to stage 1.
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p1), dim=1)
        out = self.FCSA1(out)
        out = self.backdim1(out)
        fusion1 = out

        # Final Convolution
        out = F.relu(F.interpolate(self.decoder5(fusion1), scale_factor=(2, 2), mode='bilinear'))
        final_out = self.final(out)

        if self.deep_supervision:
            input_size = x.shape[2:]
            # Deep supervision heads for p2 and p1
            out_p2 = F.interpolate(self.head_p2(p2), size=input_size, mode='bilinear', align_corners=False)
            out_p1 = F.interpolate(self.head_p1(p1), size=input_size, mode='bilinear', align_corners=False)
            return [out_p2, out_p1, final_out]  # Recommended weights: [0.4, 0.4, 1.0].

        return final_out




