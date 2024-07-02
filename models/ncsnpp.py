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
import copy

# pylint: skip-file

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name="ncsnpp")  # model0
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))
        # 当你使用register_buffer函数将一个对象注册为模型的缓冲区时，
        # 该对象将作为模型的一个属性被存储，并且在模型的前向传播过程中可以被访问。
        # 然而，与模型的参数不同，该缓冲区在训练过程中不会被优化器更新。
        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        # print("ch_mult is:",ch_mult)
        # ch_mult is: (1, 1, 2, 2, 2, 2, 2)
        self.num_res_blocks = (
            num_res_blocks
        ) = config.model.num_res_blocks  #     model.num_res_blocks = 2
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]
        # num_resolutions is len of ch_mult
        # 这个列表被赋值给了 all_resolutions 和 self.all_resolutions。
        # 所以，all_resolutions 和 self.all_resolutions 现在都包含了一个从原始图像大小开始，每次减半的一系列图像大小。
        # 例如，如果 config.data.image_size 是 512，num_resolutions 是 4，
        # 那么 all_resolutions 和 self.all_resolutions 将包含 [512, 256, 128, 64]
        self.conditional = (
            conditional
        ) = config.model.conditional  # noise-conditional   model.conditional = True
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.skip_rescale_unet = skip_rescale_unet = config.model.skip_rescale_unet
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.progressive = progressive = config.model.progressive.lower()
        self.progressive_input = (
            progressive_input
        ) = config.model.progressive_input.lower()
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]
        combine_method = config.model.progressive_combine.lower()
        combine_method_unet = config.model.progressive_combine_unet.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            assert (
                config.training.continuous
            ), "Fourier features are only used for continuous training."
            modules.append(
                layerspp.GaussianFourierProjection(
                    embedding_size=nf,
                    scale=config.model.fourier_scale,  #  model.fourier_scale = 16
                )
            )
            embed_dim = 2 * nf

        elif embedding_type == "positional":
            embed_dim = nf

        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            # (1): Linear(in_features=256, out_features=512, bias=True)
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            # print("modules[-1].weight.data is:",modules[-1].weight.data)
            nn.init.zeros_(modules[-1].bias)
            modules.append(
                nn.Linear(nf * 4, nf * 4)
            )  #  (2): Linear(in_features=512, out_features=512, bias=True)
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            # print("modules[-1].bias is :",modules[-1].bias)
            # nn.init.zeros_(modules[-1].bias) 这句代码的效果是将神经网络中最后一个模块的偏置参数初始化为全零。
            # 这通常是在模型初始化阶段进行的一步操作，以确保模型在开始训练之前有一个合理的起点。
            # 不同的初始化策略（如全零、随机初始化等）可能会对模型的训练速度和最终性能产生影响。
        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )
        # print("AttnBlock is:",AttnBlock)
        # functools.partial 函数的作用是为一个函数提供固定数量的参数，并返回一个新的函数。这个新的函数在调用时，将会使用这些固定的参数。
        # 在这句代码中，AttnBlock 是通过 functools.partial 创建的新函数，
        # 它等价于调用 layerspp.AttnBlockpp 并传入两个固定的参数 init_scale 和 skip_rescale
        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        # model.fir_kernel = [1, 3, 3, 1]
        # model.fir = True
        # model.resamp_with_conv = True

        # model.progressive = "output_skip"
        # model.progressive_input = "input_skip"
        if progressive == "output_skip":
            self.pyramid_upsample = layerspp.Upsample(
                fir=fir,
                fir_kernel=fir_kernel,
                with_conv=False,
            )
        elif progressive == "residual":
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            self.pyramid_downsample = layerspp.Downsample(
                fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        elif resblock_type == "biggan":  # model.resblock_type = "biggan"
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
                temb_dim=nf * 4,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # Downsampling block

        channels = config.data.num_channels
        unet_channels = config.data.num_channels_unet_input
        if progressive_input != "none":
            input_pyramid_ch = channels
            input_pyramid_unet_ch = unet_channels

        modules.append(conv3x3(channels, nf))
        # (3): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        modules.append(conv3x3(channels, nf))  # for unet

        hs_c = [nf]
        unet_hs_c = [nf]
        in_ch = nf
        in_unet_ch = nf

        for i_level in range(num_resolutions):  # 图像被分为num_resolutions级别
            # print("num_resolutions is:",num_resolutions)  num_resolutions is: 7
            # model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
            # Residual blocks for this resolution
            # model is stacked step by step
            for i_block in range(num_res_blocks):
                # model.num_res_blocks = 2
                out_ch = nf * ch_mult[i_level]
                unet_out_ch = nf * ch_mult[i_level]
                # if i_block == 0 and i_level == 0:
                #     print("there out_ch, unet_out_ch", out_ch, unet_out_ch)
                #     there out_ch, unet_out_ch 64 64

                modules.append(
                    ResnetBlock(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        unet_in_ch=in_unet_ch,
                        unet_out_ch=unet_out_ch,
                    )
                )
                in_ch = out_ch
                in_unet_ch = unet_out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_unet_ch))
                hs_c.append(in_ch)
                unet_hs_c.append(in_unet_ch)

            # border 1

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":  # model.resblock_type = "biggan"
                    modules.append(Downsample(in_ch=in_ch, unet_in_ch=in_unet_ch))
                else:
                    modules.append(
                        ResnetBlock(down=True, in_ch=in_ch, unet_in_ch=in_unet_ch)
                    )

                if progressive_input == "input_skip":

                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    modules.append(
                        combiner(dim1=input_pyramid_unet_ch, dim2=in_unet_ch)
                    )
                    if combine_method == "cat":  #     model.progressive_combine = "sum"
                        in_ch *= 2
                    if combine_method_unet == "cat":  # 暂时无法使用该模式
                        in_unet_ch *= 2

                elif progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(
                            in_ch=input_pyramid_ch,
                            out_ch=in_ch,
                            unet_in_ch=input_pyramid_unet_ch,
                            unet_out_ch=in_unet_ch,
                        )
                    )
                    input_pyramid_ch = in_ch
                    input_pyramid_unet_ch = in_unet_ch

                hs_c.append(in_ch)
                unet_hs_c.append(in_unet_ch)

        in_ch = hs_c[-1]
        in_unet_ch = unet_hs_c[-1]

        modules.append(ResnetBlock(in_ch=in_ch, unet_in_ch=in_unet_ch))
        modules.append(
            AttnBlock(
                channels=in_ch,
            )
        )
        modules.append(ResnetBlock(in_ch=in_ch, unet_in_ch=in_unet_ch))

        pyramid_ch = 0
        # print("hs_c is:",hs_c)
        # hs_c is:
        # [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
        # num_resolutions is: 7
        # print("num_resolutions is:",num_resolutions)

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                unet_out_ch = nf * ch_mult[i_level]
                hs_c_element_temp = hs_c.pop()
                hs_c_element_temp_unet = unet_hs_c.pop()
                modules.append(
                    ResnetBlock(
                        in_ch=in_ch + hs_c_element_temp,
                        out_ch=out_ch,
                        unet_in_ch=(in_unet_ch + hs_c_element_temp_unet),
                        unet_out_ch=unet_out_ch,
                    )
                )
                in_ch = out_ch
                in_unet_ch = unet_out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != "none":
                if i_level == num_resolutions - 1:  # the last out put exist here
                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels

                        # unet
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_unet_ch // 4, 32),
                                num_channels=in_unet_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_unet_ch, unet_channels, init_scale=init_scale)
                        )
                        pyramid_ch_unet = unet_channels

                    elif progressive == "residual":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                        # unet
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_unet_ch // 4, 32),
                                num_channels=in_unet_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(conv3x3(in_unet_ch, in_unet_ch, bias=True))
                        pyramid_ch_unet = in_unet_ch

                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:

                    if progressive == "output_skip":
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
                        )
                        pyramid_ch = channels

                        # unet
                        modules.append(
                            nn.GroupNorm(
                                num_groups=min(in_unet_ch // 4, 32),
                                num_channels=in_unet_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            conv3x3(
                                in_unet_ch,
                                unet_channels,
                                bias=True,
                                init_scale=init_scale,
                            )
                        )
                        pyramid_ch_unet = unet_channels

                    elif progressive == "residual":
                        modules.append(
                            pyramid_upsample(
                                in_ch=pyramid_ch,
                                out_ch=in_ch,
                                unet_in_ch=pyramid_ch_unet,
                                unet_out_ch=in_unet_ch,
                            )
                        )  # 暂时不管,这里还需要维护
                        pyramid_ch = in_ch
                        pyramid_ch_unet = in_unet_ch  # 未解决
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(
                        ResnetBlock(in_ch=in_ch, up=True, unet_in_ch=in_unet_ch)
                    )

        assert not hs_c  # upsample ends here , the shape of score map is made over

        if progressive != "output_skip":
            modules.append(
                nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, unet_input):
        # 推测:在训练的时候给出受扰图片以及噪声的大小指标(t),然后测试的时候就直接输入t
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if self.conditional:
            temb = modules[m_idx](
                temb
            )  #  (1): Linear(in_features=256, out_features=512, bias=True)
            m_idx += 1
            temb = modules[m_idx](
                self.act(temb)
            )  #   (2): Linear(in_features=512, out_features=512, bias=True)
            m_idx += 1
        else:
            temb = None

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0  # -1 ~ 1

        # Downsampling block
        input_pyramid = None
        input_pyramid_unet = None
        if self.progressive_input != "none":  #  model.progressive_input = "input_skip"
            input_pyramid = x
            input_pyramid_unet = unet_input
        hs = [modules[m_idx](x)]
        m_idx += 1
        hs_unet = [
            modules[m_idx](unet_input)
        ]  # (3):  Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        m_idx += 1

        # self.num_resolutions =7
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            # self.num_res_blocks = 2
            #    model.num_res_blocks = 2
            for i_block in range(self.num_res_blocks):
                h, h_unet = modules[m_idx](hs[-1], hs_unet[-1], temb)
                m_idx += 1

                if h.shape[-1] in self.attn_resolutions:  # AttnBlock(channels=in_ch)
                    h = modules[m_idx](h)
                    m_idx += 1
                if (
                    h_unet.shape[-1] in self.attn_resolutions
                ):  # AttnBlock(channels=in_ch)
                    h_unet = modules[m_idx](h_unet)
                    m_idx += 1
                hs.append(h)  # 不同迭代次数的结果的列表集合
                hs_unet.append(h_unet)  # 不同迭代次数的结果的列表集合

            # border 1  :

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":  # model.resblock_type = "biggan"
                    h, h_unet = modules[m_idx](hs[-1], hs_unet[-1])  # done
                    m_idx += 1
                else:
                    h, h_unet = modules[m_idx](
                        hs[-1], hs_unet[-1], temb, debug=1
                    )  # done
                    m_idx += 1

                if self.progressive_input == "input_skip":  # 残差连接并融合
                    input_pyramid, input_pyramid_unet = self.pyramid_downsample(
                        input_pyramid, input_pyramid_unet
                    )
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                    h_unet = modules[m_idx](input_pyramid_unet, h_unet)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid, input_pyramid_unet = modules[m_idx](
                        input_pyramid, input_pyramid_unet
                    )
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                    if self.skip_rescale_unet:
                        input_pyramid_unet = (input_pyramid_unet + h_unet) / np.sqrt(
                            2.0
                        )
                    else:
                        input_pyramid_unet = input_pyramid_unet + h_unet
                    h_unet = input_pyramid_unet

                hs.append(h)
                hs_unet.append(h_unet)
                hs_unet_reserved = hs_unet

        h = hs[-1]  # 以下三段操作不改变通道数以及大小
        h, h_unet = modules[m_idx](h, h_unet, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h, h_unet = modules[m_idx](h, h_unet, temb)
        m_idx += 1

        pyramid = None
        pyramid_unet = None
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                temp = torch.cat([h, hs.pop()], dim=1)
                temp_unet = torch.cat([h_unet, hs_unet_reserved.pop()], dim=1)
                h, h_unet = modules[m_idx](
                    temp,
                    temp_unet,
                    temb,
                )
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if (
                    i_level == self.num_resolutions - 1
                ):  # i_level 应该是从 self.num_resolutions - 1  数到 0
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1

                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1

                        #  unet
                        pyramid_unet = self.act(modules[m_idx](h_unet))
                        m_idx += 1

                        pyramid_unet = modules[m_idx](pyramid_unet)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        # unet
                        pyramid_unet = self.act(modules[m_idx](h_unet))
                        m_idx += 1
                        pyramid_unet = modules[m_idx](pyramid_unet)
                        m_idx += 1

                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:

                    if self.progressive == "output_skip":
                        pyramid, pyramid_unet = self.pyramid_upsample(
                            pyramid, pyramid_unet
                        )
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h

                        pyramid_h_unet = self.act(
                            modules[m_idx](h_unet)
                        )  # for unet :pyramid_unet
                        m_idx += 1
                        pyramid_h_unet = modules[m_idx](pyramid_h_unet)
                        m_idx += 1
                        pyramid_unet = pyramid_unet + pyramid_h_unet

                    elif self.progressive == "residual":
                        pyramid, pyramid_unet = modules[m_idx](pyramid, pyramid_unet)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                        if self.skip_rescale_unet:
                            pyramid_unet = (pyramid_unet + h) / np.sqrt(2.0)
                        else:
                            pyramid_unet = pyramid_unet + h
                        h = pyramid_unet
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h, h_unet)  # up_sample happens here
                    m_idx += 1
                else:
                    h, h_unet = modules[m_idx](h, h_unet, temb)
                    m_idx += 1

        assert not hs  # check hs is used overly
        assert not hs_unet

        # print("assert not hs  # check hs is used overly is : 3 ",h.shape)

        if self.progressive == "output_skip":
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        # print("h = modules[m_idx](h) 4", h.shape)  # 这里score map 已经基本完成了
        assert m_idx == len(modules)
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas
        return h
