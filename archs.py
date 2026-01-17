import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    except ImportError:
        # 如果都失败，使用torch的替代品
        from torch.nn import Identity as DropPath


        def to_2tuple(x):
            if isinstance(x, (list, tuple)):
                return x
            return (x, x)


        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            import torch
            torch.nn.init.trunc_normal_(tensor, mean, std, a, b)
            return tensor
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule  # MMDetection的卷积模块（已注释）
from pdb import set_trace as st  # 调试工具

from kan import KANLinear, KAN  # 导入KAN（Kolmogorov-Arnold Network）相关模块
from torch.nn import init
from FCS_attention import MultiSpectralAttentionLayer #导入FCSA模块
__all__ = ['UKAN']  # 导出UKAN模型

class SpatialAttention(nn.Module):
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
    def __init__(self, in_channels, out_channels, img_size, reduction=16):
        super(FcsAttention, self).__init__()
        c2wh = dict([(out_channels // 4, img_size // 4), (out_channels // 2, img_size // 8), (out_channels, img_size // 16)])
        # 确保 out_channels 在 c2wh 字典中是有效的键
        if out_channels not in c2wh:
            raise ValueError(f"out_channels value {out_channels} is not supported.")

        self.spatial = SpatialAttention()
        self.frequency_channel = MultiSpectralAttentionLayer(
            channel=out_channels,
            dct_h=c2wh[out_channels],
            dct_w=c2wh[out_channels],
            reduction=reduction,
            freq_sel_method='top16'
        )

    def forward(self, x):
        x = self.frequency_channel(x)
        x = self.spatial(x) * x
        return x

class ChannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelLinear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, c)
        x = self.linear(x)
        x = x.view(b, h, w, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class KANLayer(nn.Module):
    """KAN层：使用Kolmogorov-Arnold表示的前馈层"""

    def __init__(self,
                 in_features,  # 输入维度
                 hidden_features=None,  # 隐藏层维度
                 out_features=None,  # 输出维度
                 act_layer=nn.GELU,  # 激活函数（没用到）
                 drop=0.,  # Dropout概率
                 no_kan=False):  # 是否不用KAN
        super().__init__()
        out_features = out_features or in_features  # 输出特征维度，默认等于输入维度
        hidden_features = hidden_features or in_features  # 隐藏层特征维度，默认等于输入维度
        self.dim = in_features  # 保存输入维度

        # KAN层的超参数配置
        grid_size = 5  # 样条网格大小
        spline_order = 3  # 样条阶数
        scale_noise = 0.1  # 噪声缩放系数
        scale_base = 1.0  # 基础缩放系数
        scale_spline = 1.0  # 样条缩放系数
        base_activation = torch.nn.SiLU  # 基础激活函数
        grid_eps = 0.02  # 网格epsilon
        grid_range = [-1, 1]  # 网格范围

        if not no_kan:  # 如果使用KAN层
            # 创建第一个KAN全连接层
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # 创建第二个KAN全连接层
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # 创建第三个KAN全连接层
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # # TODO   # 第四个KAN层（已注释，备用）
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )

        else:  # 如果不使用KAN层，使用标准线性层
            self.fc1 = nn.Linear(in_features, hidden_features)  # 标准全连接层1
            self.fc2 = nn.Linear(hidden_features, out_features)  # 标准全连接层2
            self.fc3 = nn.Linear(hidden_features, out_features)  # 标准全连接层3

        # TODO  # 备用的标准线性层（已注释）
        # self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)  # 深度可分离卷积+BN+ReLU 1
        self.dwconv_2 = DW_bn_relu(hidden_features)  # 深度可分离卷积+BN+ReLU 2
        self.dwconv_3 = DW_bn_relu(hidden_features)  # 深度可分离卷积+BN+ReLU 3

        # # TODO  # 第四个深度卷积（已注释）
        # self.dwconv_4 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)  # Dropout层

        self.apply(self._init_weights)  # 应用权重初始化

    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):  # 如果是线性层
            trunc_normal_(m.weight, std=.02)  # 使用截断正态分布初始化权重
            if isinstance(m, nn.Linear) and m.bias is not None:  # 如果有偏置
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
        elif isinstance(m, nn.LayerNorm):  # 如果是LayerNorm层
            nn.init.constant_(m.bias, 0)  # 偏置初始化为0
            nn.init.constant_(m.weight, 1.0)  # 权重初始化为1
        elif isinstance(m, nn.Conv2d):  # 如果是卷积层
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 计算fan_out
            fan_out //= m.groups  # 考虑分组卷积
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # Kaiming初始化
            if m.bias is not None:  # 如果有偏置
                m.bias.data.zero_()  # 偏置初始化为0

    def forward(self, x, H, W):
        """前向传播"""
        # pdb.set_trace()  # 调试断点（已注释）
        B, N, C = x.shape  # B: batch size, N: token数量, C: 通道数

        x = self.fc1(x.reshape(B * N, C))  # 第一个全连接层，将(B,N,C)展平为(B*N,C)处理
        x = x.reshape(B, N, C).contiguous()  # 重塑回(B,N,C)并确保内存连续
        x = self.dwconv_1(x, H, W)  # 第一个深度卷积
        x = self.fc2(x.reshape(B * N, C))  # 第二个全连接层
        x = x.reshape(B, N, C).contiguous()  # 重塑回(B,N,C)
        x = self.dwconv_2(x, H, W)  # 第二个深度卷积
        x = self.fc3(x.reshape(B * N, C))  # 第三个全连接层
        x = x.reshape(B, N, C).contiguous()  # 重塑回(B,N,C)
        x = self.dwconv_3(x, H, W)  # 第三个深度卷积

        # # TODO  # 第四层处理（已注释）
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        return x


class KANBlock(nn.Module):
    """KAN块：包含残差连接和LayerNorm的KAN层"""

    def __init__(self,
                 dim,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 no_kan=False):
        super().__init__()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()  # Stochastic Depth，如果drop_path>0则使用，否则使用恒等映射
        self.norm2 = norm_layer(dim)  # LayerNorm层
        mlp_hidden_dim = int(dim)  # MLP隐藏层维度等于输入维度

        self.layer = KANLayer(in_features=dim,
                              hidden_features=mlp_hidden_dim,
                              act_layer=act_layer,
                              drop=drop,
                              no_kan=no_kan)  # KAN层

        self.apply(self._init_weights)  # 应用权重初始化

    def _init_weights(self, m):
        """权重初始化函数（同KANLayer）"""
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
        """前向传播：残差连接"""
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))  # 残差连接：x = x + DropPath(KANLayer(LayerNorm(x)))

        return x


class DWConv(nn.Module):
    """深度可分离卷积（Depthwise Convolution）"""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)  # 3x3深度卷积，groups=dim实现深度可分离

    def forward(self, x, H, W):
        B, N, C = x.shape  # B: batch, N: tokens, C: channels
        x = x.transpose(1, 2).view(B, C, H, W)  # 从(B,N,C)转换为(B,C,H,W)
        x = self.dwconv(x)  # 深度卷积
        x = x.flatten(2).transpose(1, 2)  # 转换回(B,N,C)格式

        return x


class DW_bn_relu(nn.Module):
    """深度可分离卷积 + BatchNorm + ReLU"""

    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)  # 3x3深度卷积
        self.bn = nn.BatchNorm2d(dim)  # BatchNorm2d
        self.relu = nn.ReLU()  # ReLU激活

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)  # (B,N,C) -> (B,C,H,W)
        x = self.dwconv(x)  # 深度卷积
        x = self.bn(x)  # BatchNorm
        x = self.relu(x)  # ReLU
        x = x.flatten(2).transpose(1, 2)  # (B,C,H,W) -> (B,N,C)

        return x


class PatchEmbed(nn.Module):
    """图像到Patch嵌入（Image to Patch Embedding）"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 确保img_size是元组格式
        patch_size = to_2tuple(patch_size)  # 确保patch_size是元组格式

        self.img_size = img_size  # 保存图像尺寸
        self.patch_size = patch_size  # 保存patch尺寸
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]  # 计算patch网格的高度和宽度
        self.num_patches = self.H * self.W  # 计算patch总数
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))  # 使用卷积实现patch嵌入
        self.norm = nn.LayerNorm(embed_dim)  # LayerNorm

        self.apply(self._init_weights)  # 应用权重初始化

    def _init_weights(self, m):
        """权重初始化函数（同上）"""
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
        x = self.proj(x)  # 卷积投影：(B,C,H,W) -> (B,embed_dim,H',W')
        _, _, H, W = x.shape  # 获取输出的高度和宽度
        x = x.flatten(2).transpose(1, 2)  # (B,embed_dim,H',W') -> (B,H'*W',embed_dim)
        x = self.norm(x)  # LayerNorm

        return x, H, W  # 返回嵌入后的tokens和空间维度


class ConvLayer(nn.Module):
    """标准卷积层：两个3x3卷积 + BN + ReLU"""

    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 第一个3x3卷积
            nn.BatchNorm2d(out_ch),  # BatchNorm
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # 第二个3x3卷积
            nn.BatchNorm2d(out_ch),  # BatchNorm
            nn.ReLU(inplace=True)  # ReLU激活
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    """解码器卷积层：两个3x3卷积 + BN + ReLU（第一个卷积通道不变）"""

    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),  # 第一个卷积保持通道数不变
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 第二个卷积改变通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UKAN(nn.Module):
    """U-KAN：基于KAN的U-Net架构用于图像分割"""

    def __init__(self, num_classes, input_channels, deep_supervision=False,
                 img_size=224, patch_size=16,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1, 1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]  # KAN层的输入维度

        # 初始的卷积编码器层 (Stage 0)
        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)
        self.encoder4 = ConvLayer(embed_dims[0], embed_dims[1])

        # 归一化层
        self.norm0 = norm_layer(embed_dims[0] // 8)
        self.norm1 = norm_layer(embed_dims[0] // 4)
        self.norm2 = norm_layer(embed_dims[0])
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        # 解码器部分的归一化层
        self.dnorm0 = norm_layer(embed_dims[0] // 8)
        self.dnorm1 = norm_layer(embed_dims[0] // 8)
        self.dnorm2 = norm_layer(embed_dims[0] // 4)
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 为每个深度生成stochastic depth rate

        # Encoder 的 KAN Block 列表
        self.block01 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 8,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block12 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 4,
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.block23 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer
        )])

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[3], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[4], norm_layer=norm_layer
        )])

        # Decoder 的 KAN Block 列表
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[4], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[3], norm_layer=norm_layer
        )])

        self.dblock23 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 4,
            drop=drop_rate, drop_path=dpr[2], norm_layer=norm_layer
        )])

        self.dblock12 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 8,
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.dblock01 = nn.ModuleList([KANBlock(
            dim=embed_dims[0] // 8,
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        # 解码器卷积层
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        # 1x1 卷积用于调整上采样后的通道数
        self.upsample1 = nn.Conv2d(in_channels=embed_dims[2], out_channels=embed_dims[1], kernel_size=1)
        self.upsample2 = nn.Conv2d(in_channels=embed_dims[1], out_channels=embed_dims[0], kernel_size=1)
        self.upsample3 = nn.Conv2d(in_channels=embed_dims[0], out_channels=embed_dims[0] // 4, kernel_size=1)
        self.upsample4 = nn.Conv2d(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 8, kernel_size=1)
        self.upsample5 = nn.Conv2d(in_channels=embed_dims[0] // 8, out_channels=embed_dims[0] // 8, kernel_size=1)

        # 融合阶段使用的 FCSA 注意力模块 (Fusion Stage)
        self.FCSA1 = FcsAttention(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 4, img_size=img_size)
        self.FCSA2 = FcsAttention(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 2, img_size=img_size)
        self.FCSA3 = FcsAttention(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0] * 2, img_size=img_size)
        self.FCSA4 = FcsAttention(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1] * 2, img_size=img_size)

        # Skip Connection 阶段使用的 FCSA 注意力模块
        self.FCSA1s = FcsAttention(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 4, img_size=img_size)
        self.FCSA2s = FcsAttention(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 2, img_size=img_size)
        self.FCSA3s = FcsAttention(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0] * 2, img_size=img_size)
        self.FCSA4s = FcsAttention(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1] * 2, img_size=img_size)

        # 通道降维层 (s后缀对应 Skip Connection 处理)
        self.backdim1s = ChannelLinear(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 8)
        self.backdim2s = ChannelLinear(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 4)
        self.backdim3s = ChannelLinear(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0])
        self.backdim4s = ChannelLinear(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1])

        # 通道降维层 (对应 Fusion 阶段)
        self.backdim1 = ChannelLinear(in_channels=embed_dims[0] // 4, out_channels=embed_dims[0] // 8)
        self.backdim2 = ChannelLinear(in_channels=embed_dims[0] // 2, out_channels=embed_dims[0] // 4)
        self.backdim3 = ChannelLinear(in_channels=embed_dims[0] * 2, out_channels=embed_dims[0])
        self.backdim4 = ChannelLinear(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1])

        # 最终输出层
        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        """前向传播"""
        B = x.shape[0]

        ### Stage 1 (Encoder)
        out, H, W = self.patch_embed0(x)
        for i, blk in enumerate(self.block01):
            out = blk(out, H, W)
        out = self.norm0(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t1 = out  # 保存 skip connection 特征 t1

        ### Stage 2 (Encoder)
        out, H, W = self.patch_embed1(out)
        for i, blk in enumerate(self.block12):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out  # 保存 skip connection 特征 t2

        ### Stage 3 (Encoder)
        out, H, W = self.patch_embed2(out)
        for i, blk in enumerate(self.block23):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out  # 保存 skip connection 特征 t3

        ### Stage 4 (Encoder)
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out  # 保存 skip connection 特征 t4

        ### Bottleneck (瓶颈层)
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p5 = out  # 瓶颈层特征 p5

        ### Decoder Stage 4 (第一阶段解码)
        # 上采样并进行卷积
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))

        # 与 t4 进行拼接 (Skip Connection)
        # out = torch.add(out, t4)
        out = torch.cat((out, t4), dim=1)
        # 注意力机制 + 通道调整
        out = self.FCSA4s(out)
        out = self.backdim4s(out)

        # 通过 KAN Block 进行特征处理
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Decoder Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p4 = out  # 获得 p4 特征

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        # 与 t3 拼接
        # out = torch.add(out, t3)
        out = torch.cat((out, t3), dim=1)
        out = self.FCSA3s(out)
        out = self.backdim3s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p3 = out  # 获得 p3 特征

        ### Decoder Stage 2
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        # 与 t2 拼接
        # out = torch.add(out, t2)
        out = torch.cat((out, t2), dim=1)
        out = self.FCSA2s(out)
        out = self.backdim2s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock23):
            out = blk(out, H, W)

        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p2 = out  # 获得 p2 特征

        ### Decoder Stage 1
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        # 与 t1 拼接
        # out = torch.add(out, t1)
        out = torch.cat((out, t1), dim=1)
        out = self.FCSA1s(out)
        out = self.backdim1s(out)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock12):
            out = blk(out, H, W)

        out = self.dnorm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        p1 = out  # 获得 p1 特征

        ### Fusion / Reconstruction Path (第二阶段解码/融合)
        # 将 p5-p1 特征逐级上采样并融合
        out = self.upsample1(p5)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p4), dim=1)
        out = self.FCSA4(out)
        out = self.backdim4(out)

        out = self.upsample2(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p3), dim=1)
        out = self.FCSA3(out)
        out = self.backdim3(out)

        out = self.upsample3(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p2), dim=1)
        out = self.FCSA2(out)
        out = self.backdim2(out)

        out = self.upsample4(out)
        out = F.relu(F.interpolate(out, scale_factor=(2, 2), mode='nearest'))
        out = torch.cat((out, p1), dim=1)
        out = self.FCSA1(out)
        out = self.backdim1(out)

        # 最终处理和分类
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)
