"""Model definitions for U-KAN and optional refinement heads.

This module contains the U-KAN backbone, attention blocks, KAN token blocks,
and optional post-decoder refinement modules used during training and testing.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
import warnings

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
        # Kept for API compatibility with earlier variants.
        _ = in_channels
        _ = img_size

        self.out_channels = out_channels
        self.left_channels = out_channels // 2
        self.right_channels = out_channels - self.left_channels
        left_hidden = max(self.left_channels // reduction, 4)
        right_hidden = max(self.right_channels // reduction, 4)

        # Decoupled gating for upsample branch and skip branch features.
        self.channel_left = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.left_channels, left_hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(left_hidden, self.left_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.channel_right = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.right_channels, right_hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(right_hidden, self.right_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.cross_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial = SpatialAttention()

        # Fixed EM priors: gradient and Laplacian emphasize scattering boundaries.
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        lap = torch.tensor([[0.0, -1.0, 0.0],
                            [-1.0, 4.0, -1.0],
                            [0.0, -1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('em_sobel_x', sobel_x.repeat(out_channels, 1, 1, 1), persistent=False)
        self.register_buffer('em_sobel_y', sobel_y.repeat(out_channels, 1, 1, 1), persistent=False)
        self.register_buffer('em_lap', lap.repeat(out_channels, 1, 1, 1), persistent=False)
        # Learnable multi-scale fusion weights for d=1/2/3 priors.
        self.em_scale_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        # Sparse-focus gate on high-scattering regions.
        self.focus_tau = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.focus_gamma = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.em_blend = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    @staticmethod
    def _normalize_map(v):
        return v / v.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    def _em_prior(self, x):
        kx = self.em_sobel_x.to(dtype=x.dtype)
        ky = self.em_sobel_y.to(dtype=x.dtype)
        kl = self.em_lap.to(dtype=x.dtype)

        em_maps = []
        for dilation in (1, 2, 3):
            gx = F.conv2d(x, kx, padding=dilation, dilation=dilation, groups=self.out_channels)
            gy = F.conv2d(x, ky, padding=dilation, dilation=dilation, groups=self.out_channels)
            lap = F.conv2d(x, kl, padding=dilation, dilation=dilation, groups=self.out_channels).abs()
            edge = torch.sqrt(gx * gx + gy * gy + 1e-12)
            em = (edge + lap).mean(dim=1, keepdim=True)
            em_maps.append(self._normalize_map(em))

        scale_w = F.softmax(self.em_scale_logits, dim=0).to(dtype=x.dtype)
        em_multi = scale_w[0] * em_maps[0] + scale_w[1] * em_maps[1] + scale_w[2] * em_maps[2]

        tau = torch.clamp(self.focus_tau, min=0.0, max=1.0).to(dtype=x.dtype)
        gamma = torch.clamp(self.focus_gamma, min=1.0, max=30.0).to(dtype=x.dtype)
        focus = torch.sigmoid(gamma * (em_multi - tau))
        return self._normalize_map(em_multi * focus)

    def forward(self, x):
        # 1) Decoupled branch gating.
        x_left = x[:, :self.left_channels]
        x_right = x[:, self.left_channels:]
        x_left = x_left * self.channel_left(x_left)
        x_right = x_right * self.channel_right(x_right)
        x_decoupled = torch.cat([x_left, x_right], dim=1)
        x_decoupled = x_decoupled * self.cross_gate(x_decoupled)

        # 2) Spatial gate + 3) EM sparse-focus prior gate.
        spatial_gate = self.spatial(x_decoupled)
        em_gate = self._em_prior(x_decoupled)
        alpha = torch.clamp(self.em_blend, min=0.0, max=1.0).to(dtype=x.dtype)
        gate = (1.0 - alpha) * spatial_gate + alpha * em_gate
        return gate * x_decoupled


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
        B, _, C = x.shape
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
        # Reserved for interface consistency.
        _ = act_layer
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
        # Reserved for interface compatibility.
        _ = patch_size

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Base channel schedule:
        # Level 0: embed_dims[0] // 8
        # Level 1: embed_dims[0] // 4
        # Level 2: embed_dims[0] (bottleneck)
        base_dim = embed_dims[0]
        self.use_edge_residual_refine = kwargs.get('use_edge_residual_refine', False)
        self.use_edge_multiscale = kwargs.get('use_edge_multiscale', False)
        self.use_edge_sparse_focus = kwargs.get('use_edge_sparse_focus', False)
        self.use_edge_center_boost = kwargs.get('use_edge_center_boost', False)
        self.use_dual_ri_refine = kwargs.get('use_dual_ri_refine', False)
        self.use_detail_skip_refine = kwargs.get('use_detail_skip_refine', False)
        self.use_fourier_refine = kwargs.get('use_fourier_refine', False)
        self.fourier_use_fft = kwargs.get('fourier_use_fft', True)

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
            [KANBlock(dim=base_dim // 8, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan)])
        # Stage 2
        self.block12 = nn.ModuleList(
            [KANBlock(dim=base_dim // 4, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan)])
        # Stage 3 (Bottleneck)
        self.block23 = nn.ModuleList([KANBlock(dim=base_dim, drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer,
                                               no_kan=no_kan)])

        # --- KAN Blocks (Decoder) ---
        # Decode Stage 2
        self.dblock23 = nn.ModuleList(
            [KANBlock(dim=base_dim // 4, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan)])
        # Decode Stage 1
        self.dblock12 = nn.ModuleList(
            [KANBlock(dim=base_dim // 8, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan)])

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

        # Optional real/imag split tail refinement on top of shared decoder feature.
        if self.use_dual_ri_refine and num_classes == 2 and input_channels >= 2:
            ri_mid = int(kwargs.get('ri_refine_mid', max(base_dim // 4, 32)))
            ri_in = (base_dim // 8) + 2  # [shared feat, input single-part, base pred single-part]
            self.ri_refine_real = nn.Sequential(
                nn.Conv2d(ri_in, ri_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ri_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(ri_mid, ri_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ri_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(ri_mid, 1, kernel_size=1, bias=True),
            )
            self.ri_refine_imag = nn.Sequential(
                nn.Conv2d(ri_in, ri_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ri_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(ri_mid, ri_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ri_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(ri_mid, 1, kernel_size=1, bias=True),
            )
            ri_scale = float(kwargs.get('ri_refine_scale', 0.08))
            self.ri_refine_scale = nn.Parameter(torch.tensor(ri_scale, dtype=torch.float32))

        # Optional Fourier-domain refinement on final full-resolution features.
        if self.use_fourier_refine:
            freq_mid = int(kwargs.get('fourier_refine_mid', max(base_dim // 4, 32)))
            freq_in = (base_dim // 8) * 2
            self.fourier_refine_net = nn.Sequential(
                nn.Conv2d(freq_in, freq_mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(freq_mid),
                nn.GELU(),
                nn.Conv2d(freq_mid, freq_in, kernel_size=1, bias=False),
            )
            self.fourier_fallback_net = nn.Sequential(
                nn.Conv2d(base_dim // 8, freq_mid, kernel_size=1, bias=False),
                nn.BatchNorm2d(freq_mid),
                nn.GELU(),
                nn.Conv2d(freq_mid, base_dim // 8, kernel_size=1, bias=False),
            )
            freq_scale = float(kwargs.get('fourier_refine_scale', 0.1))
            self.fourier_refine_scale = nn.Parameter(torch.full((1, base_dim // 8, 1, 1), freq_scale, dtype=torch.float32))
            self._fourier_warned = False

        # Optional edge-aware residual refinement head.
        if self.use_edge_residual_refine:
            edge_mid = int(kwargs.get('edge_refine_mid', max(base_dim // 4, 32)))
            edge_in = input_channels * 2 + num_classes  # [input, sobel(input), base_pred]
            self.edge_refine_feat = nn.Sequential(
                nn.Conv2d(edge_in, edge_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(edge_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_mid, edge_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(edge_mid),
                nn.ReLU(inplace=True),
            )

            if self.use_edge_multiscale:
                self.edge_ms_b1 = nn.Sequential(
                    nn.Conv2d(edge_mid, edge_mid, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(edge_mid),
                    nn.ReLU(inplace=True),
                )
                self.edge_ms_b2 = nn.Sequential(
                    nn.Conv2d(edge_mid, edge_mid, kernel_size=3, padding=2, dilation=2, bias=False),
                    nn.BatchNorm2d(edge_mid),
                    nn.ReLU(inplace=True),
                )
                self.edge_ms_b3 = nn.Sequential(
                    nn.Conv2d(edge_mid, edge_mid, kernel_size=3, padding=3, dilation=3, bias=False),
                    nn.BatchNorm2d(edge_mid),
                    nn.ReLU(inplace=True),
                )
                self.edge_ms_fuse = nn.Sequential(
                    nn.Conv2d(edge_mid * 3, edge_mid, kernel_size=1, bias=False),
                    nn.BatchNorm2d(edge_mid),
                    nn.ReLU(inplace=True),
                )

            if num_classes == 2:
                # Decouple real/imag residual prediction to reduce branch interference.
                self.edge_delta_real = nn.Conv2d(edge_mid, 1, kernel_size=1, bias=True)
                self.edge_delta_imag = nn.Conv2d(edge_mid, 1, kernel_size=1, bias=True)
                self.edge_gate_real = nn.Conv2d(edge_mid, 1, kernel_size=1, bias=True)
                self.edge_gate_imag = nn.Conv2d(edge_mid, 1, kernel_size=1, bias=True)
            else:
                self.edge_delta = nn.Conv2d(edge_mid, num_classes, kernel_size=1, bias=True)
                self.edge_gate = nn.Conv2d(edge_mid, num_classes, kernel_size=1, bias=True)

            init_scale = float(kwargs.get('edge_refine_scale', 0.2))
            self.edge_refine_scale = nn.Parameter(torch.full((1, num_classes, 1, 1), init_scale, dtype=torch.float32))

            if self.use_edge_center_boost:
                center_boost = float(kwargs.get('edge_center_boost', 0.6))
                center_sigma = float(kwargs.get('edge_center_sigma', 0.45))
                self.edge_center_boost = nn.Parameter(torch.tensor(center_boost, dtype=torch.float32))
                self.edge_center_sigma = center_sigma

            if self.use_edge_sparse_focus:
                focus_tau = float(kwargs.get('edge_focus_tau', 0.30))
                focus_gamma = float(kwargs.get('edge_focus_gamma', 10.0))
                self.edge_focus_tau = nn.Parameter(torch.tensor(focus_tau, dtype=torch.float32))
                self.edge_focus_gamma = focus_gamma

            # Fixed Sobel kernels used to expose high-frequency cues.
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                    [-2.0, 0.0, 2.0],
                                    [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                    [0.0, 0.0, 0.0],
                                    [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x, persistent=False)
            self.register_buffer('sobel_y', sobel_y, persistent=False)

        # Optional detail skip refinement head for high-frequency recovery.
        if self.use_detail_skip_refine:
            detail_mid = int(kwargs.get('detail_refine_mid', max(base_dim // 2, 64)))
            detail_in = (base_dim // 8) + num_classes + input_channels
            self.detail_refine_feat = nn.Sequential(
                nn.Conv2d(detail_in, detail_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(detail_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(detail_mid, detail_mid, kernel_size=3, padding=1, groups=detail_mid, bias=False),
                nn.BatchNorm2d(detail_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(detail_mid, detail_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(detail_mid),
                nn.ReLU(inplace=True),
            )

            if num_classes == 2:
                self.detail_delta_real = nn.Conv2d(detail_mid, 1, kernel_size=1, bias=True)
                self.detail_delta_imag = nn.Conv2d(detail_mid, 1, kernel_size=1, bias=True)
                self.detail_gate_real = nn.Conv2d(detail_mid, 1, kernel_size=1, bias=True)
                self.detail_gate_imag = nn.Conv2d(detail_mid, 1, kernel_size=1, bias=True)
            else:
                self.detail_delta = nn.Conv2d(detail_mid, num_classes, kernel_size=1, bias=True)
                self.detail_gate = nn.Conv2d(detail_mid, num_classes, kernel_size=1, bias=True)

            detail_scale = float(kwargs.get('detail_refine_scale', 0.1))
            self.detail_refine_scale = nn.Parameter(torch.full((1, num_classes, 1, 1), detail_scale, dtype=torch.float32))

            lap = torch.tensor([[0.0, -1.0, 0.0],
                                [-1.0, 4.0, -1.0],
                                [0.0, -1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('lap_kernel', lap, persistent=False)

        # --- Deep Supervision Heads ---
        if self.deep_supervision:
            # Intermediate supervision heads for p2 and p1.
            self.head_p2 = nn.Conv2d(base_dim // 4, num_classes, kernel_size=1)
            self.head_p1 = nn.Conv2d(base_dim // 8, num_classes, kernel_size=1)

    def _sobel_grad_mag(self, x):
        c = x.shape[1]
        kx = self.sobel_x.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        ky = self.sobel_y.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def _laplacian(self, x):
        c = x.shape[1]
        k = self.lap_kernel.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        return F.conv2d(x, k, padding=1, groups=c)

    def _apply_fourier_refine(self, feat):
        # Run FFT refinement in FP32 for numerical stability and cast back.
        feat_dtype = feat.dtype
        feat_f = feat.float().contiguous()
        _, _, h, w = feat_f.shape
        delta_spatial = None

        if self.fourier_use_fft:
            try:
                x_fft = torch.fft.rfft2(feat_f, norm='ortho')
                freq = torch.cat([x_fft.real, x_fft.imag], dim=1)
                delta_freq = self.fourier_refine_net(freq)
                d_real, d_imag = torch.chunk(delta_freq, 2, dim=1)
                delta_complex = torch.complex(d_real, d_imag)
                delta_spatial = torch.fft.irfft2(delta_complex, s=(h, w), norm='ortho')
            except RuntimeError as e:
                if 'CUFFT' in str(e).upper():
                    if not self._fourier_warned:
                        warnings.warn(
                            "cuFFT failed in fourier_refine; fallback to spatial refine.",
                            RuntimeWarning
                        )
                        self._fourier_warned = True
                else:
                    raise

        if delta_spatial is None:
            delta_spatial = self.fourier_fallback_net(feat_f)

        scale = torch.clamp(self.fourier_refine_scale, min=0.0, max=1.0).to(dtype=delta_spatial.dtype)
        out = feat_f + scale * delta_spatial
        return out.to(dtype=feat_dtype)

    def forward(self, x):
        B = x.shape[0]

        # ================= Encoder Path (3 Stages) =================
        # Stage 1
        out, H, W = self.patch_embed0(x)
        for blk in self.block01: out = blk(out, H, W)
        out = self.norm0(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t1 = out  # Level 0 feature

        # Stage 2
        out, H, W = self.patch_embed1(out)
        for blk in self.block12: out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out  # Level 1 feature

        # Stage 3 (Bottleneck)
        out, H, W = self.patch_embed2(out)
        for blk in self.block23: out = blk(out, H, W)
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
        for blk in self.dblock23: out = blk(out, H, W)  # KAN Block

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
        for blk in self.dblock12: out = blk(out, H, W)

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
        if self.use_fourier_refine:
            out = self._apply_fourier_refine(out)
        final_out = self.final(out)
        if self.use_dual_ri_refine and self.num_classes == 2 and x.shape[1] >= 2:
            x_real = x[:, 0:1]
            x_imag = x[:, 1:2]
            p_real = final_out[:, 0:1]
            p_imag = final_out[:, 1:2]
            ri_in_real = torch.cat([out, x_real, p_real], dim=1)
            ri_in_imag = torch.cat([out, x_imag, p_imag], dim=1)
            d_real = self.ri_refine_real(ri_in_real)
            d_imag = self.ri_refine_imag(ri_in_imag)
            ri_scale = torch.clamp(self.ri_refine_scale, min=0.0, max=1.0).to(dtype=final_out.dtype)
            final_out = torch.cat([p_real + ri_scale * d_real, p_imag + ri_scale * d_imag], dim=1)
        if self.use_edge_residual_refine:
            grad_x = self._sobel_grad_mag(x)
            edge_in = torch.cat([x, grad_x, final_out], dim=1)
            feat = self.edge_refine_feat(edge_in)
            if self.use_edge_multiscale:
                f1 = self.edge_ms_b1(feat)
                f2 = self.edge_ms_b2(feat)
                f3 = self.edge_ms_b3(feat)
                feat = self.edge_ms_fuse(torch.cat([f1, f2, f3], dim=1))
            if self.num_classes == 2:
                delta = torch.cat([self.edge_delta_real(feat), self.edge_delta_imag(feat)], dim=1)
                gate = torch.sigmoid(torch.cat([self.edge_gate_real(feat), self.edge_gate_imag(feat)], dim=1))
            else:
                delta = self.edge_delta(feat)
                gate = torch.sigmoid(self.edge_gate(feat))
            scale = torch.clamp(self.edge_refine_scale, min=0.0, max=1.0).to(dtype=final_out.dtype)
            edge_strength = grad_x.mean(dim=1, keepdim=True)
            edge_strength = edge_strength / edge_strength.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            focus = 1.0
            if self.use_edge_sparse_focus:
                tau = torch.clamp(self.edge_focus_tau, min=0.0, max=1.0).to(dtype=edge_strength.dtype)
                focus = torch.sigmoid(self.edge_focus_gamma * (edge_strength - tau))
            if self.use_edge_center_boost:
                # Hard-error locations are mostly center-edge pixels in this dataset.
                # Boost correction where both edge strength and center prior are high.
                h, w = edge_strength.shape[-2], edge_strength.shape[-1]
                yy = torch.linspace(-1.0, 1.0, h, device=edge_strength.device, dtype=edge_strength.dtype).view(1, 1, h, 1)
                xx = torch.linspace(-1.0, 1.0, w, device=edge_strength.device, dtype=edge_strength.dtype).view(1, 1, 1, w)
                sigma = max(self.edge_center_sigma, 1e-3)
                center_prior = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
                boost_gain = torch.clamp(self.edge_center_boost, min=0.0, max=2.0).to(dtype=edge_strength.dtype)
                boost = 1.0 + boost_gain * center_prior * edge_strength
                final_out = final_out + scale * boost * focus * gate * delta
            else:
                final_out = final_out + scale * focus * gate * delta

        if self.use_detail_skip_refine:
            lap_x = self._laplacian(x)
            detail_in = torch.cat([out, final_out, lap_x], dim=1)
            dfeat = self.detail_refine_feat(detail_in)
            if self.num_classes == 2:
                ddelta = torch.cat([self.detail_delta_real(dfeat), self.detail_delta_imag(dfeat)], dim=1)
                dgate = torch.sigmoid(torch.cat([self.detail_gate_real(dfeat), self.detail_gate_imag(dfeat)], dim=1))
            else:
                ddelta = self.detail_delta(dfeat)
                dgate = torch.sigmoid(self.detail_gate(dfeat))
            dscale = torch.clamp(self.detail_refine_scale, min=0.0, max=1.0).to(dtype=final_out.dtype)
            final_out = final_out + dscale * dgate * ddelta

        if self.deep_supervision:
            input_size = x.shape[2:]
            # Deep supervision heads for p2 and p1
            out_p2 = F.interpolate(self.head_p2(p2), size=input_size, mode='bilinear', align_corners=False)
            out_p1 = F.interpolate(self.head_p1(p1), size=input_size, mode='bilinear', align_corners=False)
            return [out_p2, out_p1, final_out]  # Recommended weights: [0.4, 0.4, 1.0].

        return final_out




