# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        # torch.randn(embedding_size) 这行代码会返回一个包含 embedding_size 个元素的张量（tensor），
        # 这些元素都是从标准正态分布（均值为0，标准差为1）中随机抽取的。
        # 因此，返回的张量中的每个元素都是浮点数，其值大致在 -1 到 1 之间，但也可能超出这个范围，
        # 因为正态分布是一个连续分布，理论上可以取到任何实数值，尽管取到远离 -1 和 1 的值的概率较小。
        # print("self.W is:", self.W)
        """
      tensor([-20.3582,  -9.5820,   8.3667,  12.2760, -16.7716,   2.0710, -14.9407,
          1.9399, -22.1855,  29.3103, -30.4583,  -2.6062,  19.4472,  -9.0339,
        -11.7150,  -3.3779,  -0.9773,  10.0466, -20.0382,   2.8213,   8.4212,
          7.3640,  -1.1076, -12.5631,   1.8297,   1.5342,   3.2045,   0.7270,
        -10.2173,  13.9922,  -6.4423, -22.1875,  -7.1333,   8.6023,  28.2157,
        -24.3800,  -3.9272,   0.2005,   6.0048, -30.1014,  10.2266,  -2.3752,
          9.6486,   9.8218,  11.7152,   3.0696, -28.2029,  -5.9392,  -1.7674,
        -18.7196, -17.3064,  10.9320,  20.3064,  14.9440,  -8.4913,  -4.4917,
         30.9084,   8.4070,  -0.1661,   5.1486,  -5.2341,   3.8302,  37.5200,
        -30.9236])

    """

    def forward(self, x):
        # print("x is:", x, "\n", x.shape)
        # print("x[:, None] is :", x[:, None], "\n", x[:, None].shape)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        # 在表达式 self.W[None, :] 中，self.W 可能是一个一维数组（例如NumPy数组或TensorFlow张量）。
        # 这个表达式的目的是创建一个新的二维数组或张量，
        # 其中 self.W 被扩展为一个只有一行的二维数组或张量
        # print("x_proj is:", x_proj, "\n", x_proj.shape)
        # print(
        #     "torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) is:",
        #     torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1),
        #     "\n",
        #     torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).shape,
        # )
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    # GaussianFourierProjection 模块能够将输入 x 转换为一个更高维的特征表示，该表示同时考虑了输入数据的幅度和相位信息。
    # 这在处理具有周期性或频率特性的数据时可能非常有用，例如在信号处理、时间序列分析或音频处理等领域。


class Unet_downsample(nn.Module):
    def __init__(self, unet_in, debug=0):
        super().__init__()
        if debug == 1:
            print("################ ")
            print("self.f = unet_in", unet_in)
        self.unet_in = unet_in
        self.devide = 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.unet_in,
                out_channels=self.unet_in,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.unet_in,
                out_channels=self.unet_in,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.max = nn.MaxPool2d(kernel_size=self.devide, stride=self.devide)

    def forward(self, x, debug=0):
        x = self.conv_block(x)
        x = self.max(x)
        return x


class Unet_upsample(nn.Module):
    def __init__(self, unet_in, debug=0):
        super().__init__()
        self.unet_in = unet_in
        self.up = nn.ConvTranspose2d(
            in_channels=self.unet_in, out_channels=self.unet_in, kernel_size=2, stride=2
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.unet_in,
                out_channels=self.unet_in,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.unet_in,
                out_channels=self.unet_in,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, unet_input, debug=0):
        # print("here:", unet_input)
        x = self.conv_up(unet_input)
        x = self.up(x)

        return x


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method="sum", debug=0):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method
        self.dims = [dim1, dim2]

    def forward(self, x, y, debug=0, **arg):
        # print("Combine come in as: x:", x.shape, y.shape)
        if debug == 1:
            print(
                "#############",
                x.shape,
                y.shape,
            )
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        elif self.method == "sum":
            return h + y

        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AttnBlockpp(nn.Module):
    """
    Channel-wise self-attention block. Modified from DDPM.
    AttnBlockpp come in as: torch.Size([1, 256, 16, 16])
    come out as torch.Size([1, 256, 16, 16])
    """

    def __init__(self, channels, skip_rescale=False, init_scale=0.0, debug=0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x, debug=0):

        # print("AttnBlockpp come in as:", x.shape)
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:  # model.skip_rescale = True
            # print("come out as", (x + h).shape)
            return x + h

        else:
            # print("come out as", ((x + h) / np.sqrt(2.0)).shape)
            return (x + h) / np.sqrt(2.0)


class Upsample(nn.Module):
    """
    Upsample come in as: x: torch.Size([1, 1, 8, 8])
    come out as: torch.Size([1, 1, 16, 16])
    """

    def __init__(
        self,
        in_ch=None,
        out_ch=None,
        with_conv=False,
        fir=False,
        unet_in_ch=None,
        unet_out_ch=None,
        fir_kernel=(1, 3, 3, 1),
        debug=0,
    ):

        super().__init__()
        unet_out_ch = unet_out_ch if unet_out_ch else unet_in_ch
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.debug = debug
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x, unet_input, debug=0):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), "nearest")
            unet_output = F.interpolate(unet_input, (H * 2, W * 2), "nearest")
            if self.with_conv:
                h = self.Conv_0(h)
                unet_output = self.Conv_0(unet_output)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
                unet_output = up_or_down_sampling.upsample_2d(
                    unet_input, self.fir_kernel, factor=2
                )
            else:
                # print(x, unet_input)
                h = self.Conv2d_0(x)
                unet_output = self.Conv2d_0(unet_input)
                # print(h, unet_output)
        return h, unet_output


class Downsample(nn.Module):
    """
    Downsample come in as: x: torch.Size([1, 1, 256, 256])
    Downsample come out as: torch.Size([1, 1, 128, 128])
    """

    def __init__(
        self,
        in_ch=None,
        out_ch=None,
        with_conv=False,
        fir=False,
        unet_in_ch=None,
        unet_out_ch=None,
        fir_kernel=(1, 3, 3, 1),
        debug=0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.unet_in_ch = unet_in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.unet_out_ch = unet_out_ch if unet_in_ch else unet_in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:

                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )

        self.conv_unet = conv1x1(self.unet_in_ch, self.unet_out_ch)
        self.unet_downsample = Unet_downsample(unet_in=self.unet_out_ch)
        self.combine = Combine(dim1=self.unet_out_ch, dim2=out_ch)

        # downsample 的方法需要多尝试几种:
        # 备选方法:
        #        up_or_down_sampling.Conv2d
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.debug = debug

    def forward(self, x, Unet_in_data, debug=0):
        unet_out_data = self.conv_unet(Unet_in_data)
        unet_out_data = self.unet_downsample(unet_out_data)

        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
                # x_unet = F.pad(Unet_in_data, (0, 1, 0, 1))
            else:
                x = F.avg_pool2d(x, 2, stride=2)
                # x_unet = F.avg_pool2d(Unet_in_data, 2, stride=2)
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
                # x_unet = up_or_down_sampling.downsample_2d(Unet_in_data, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)
                # x_unet = self.Conv2d_0(Unet_in_data)

        # print("Downsample come out as:", x.shape)

        # try to migrate the information here ljb 2024/3/3
        x = self.combine(unet_out_data, x)

        return x, unet_out_data


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
        in_unet_ch=False,
        unet_out_ch=False,
        debug=0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None, debug=0):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):  # ResnetBlockBigGANpp  can be used to upsample
    """
    ResnetBlockBigGANpp come in as: x: torch.Size([1, 128, 256, 256])
    come out as: torch.Size([1, 128, 256, 256])
    """

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        unet_in_ch=None,
        unet_out_ch=None,
        skip_rescale=True,
        init_scale=0.0,
        debug=0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        unet_out_ch = unet_out_ch if unet_out_ch else unet_in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)
        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.unet_in_ch = unet_in_ch
        self.unet_out_ch = unet_out_ch

        if unet_in_ch != unet_out_ch or up or down:
            self.Conv_2_unet = conv1x1(unet_in_ch, unet_out_ch)

        if self.down:
            self.unet_sample = Unet_downsample(unet_in=self.unet_in_ch)
        elif self.up:
            self.unet_sample = Unet_upsample(unet_in=self.unet_in_ch)

        self.combine = Combine(dim1=self.unet_out_ch, dim2=out_ch)
        # downsample 的方法需要多尝试几种:
        # 备选方法:
        #        up_or_down_sampling.Conv2d

    def forward(self, x, unet_input, temb=None, unet_residual=None, debug=0):
        if self.up:
            unet_input = self.unet_sample(unet_input)
        if self.down:
            unet_input = self.unet_sample(unet_input)
        h = self.act(self.GroupNorm_0(x))
        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:

            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)
        if self.unet_in_ch != self.unet_out_ch or self.up or self.down:
            unet_input = self.Conv_2_unet(unet_input)

        x = self.combine(unet_input, (x + h))
        unet_out_data = unet_input
        if not self.skip_rescale:
            return x, unet_out_data
        else:
            return x / np.sqrt(2.0), unet_out_data
