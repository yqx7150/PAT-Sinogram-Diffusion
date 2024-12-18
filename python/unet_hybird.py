import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import sigmoid


class UNet(nn.Module):
    def __init__(self, f=32):  ##最开始的时候需要输入参数
        super(UNet, self).__init__()
        self.f = f
        # Conv block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=self.f, kernel_size=3, padding=1, stride=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f,
                out_channels=self.f,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 所以最开始输入什么大小的正方形都是可以的，
        # 然后这个网络还可以用来测试不同大小的正方形

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(
                in_channels=f,
                out_channels=self.f * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 2,
                out_channels=self.f * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 2,
                out_channels=self.f * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 4,
                out_channels=self.f * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 4,
                out_channels=self.f * 8,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 8,
                out_channels=self.f * 8,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 8,
                out_channels=self.f * 16,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 16,
                out_channels=self.f * 16,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Up 1
        self.up_1 = nn.ConvTranspose2d(
            in_channels=self.f * 16, out_channels=self.f * 8, kernel_size=2, stride=2
        )
        # 在 PyTorch 中，ConvTranspose2d 是一个二维转置卷积层（有时也称为反卷积层）。转置卷积（有时也称为反卷积或逆卷积）并不是卷积的真正逆操作，但它通常用于实现上采样或增大特征图的大小。
        # 在深度学习，特别是图像分割、超分辨率和其他需要上采样的任务中，ConvTranspose2d 是一种非常有用的工具。
        #
        # 具体来说，ConvTranspose2d 的功能包括：
        #
        #     上采样：与常规的 Conv2d 层（用于下采样）不同，ConvTranspose2d 层用于上采样。当你想要将特征图的大小从较小的尺寸恢复到较大的尺寸时，这个层很有用。
        #     学习上采样权重：与简单的上采样方法（如双线性插值或最近邻插值）不同，ConvTranspose2d 学习如何最好地上采样。这意味着它可以学习如何根据输入数据和任务需求来优化上采样过程。
        #     在编解码器结构中的应用：ConvTranspose2d 在编解码器（encoder-decoder）结构中特别有用，这种结构通常用于图像分割任务。在这种结构中，编码器部分通过一系列卷积和池化操作减小了特征图的大小，而解码器部分则使用 ConvTranspose2d 层来恢复特征图的大小，并输出与输入图像相同大小的分割图。
        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 16,
                out_channels=self.f * 8,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 8,
                out_channels=self.f * 8,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(
            in_channels=self.f * 8, out_channels=self.f * 4, kernel_size=2, stride=2
        )

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 8,
                out_channels=self.f * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 4,
                out_channels=self.f * 4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(
            in_channels=self.f * 4, out_channels=self.f * 2, kernel_size=2, stride=2
        )

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 4,
                out_channels=self.f * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f * 2,
                out_channels=self.f * 2,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(
            in_channels=self.f * 2, out_channels=self.f, kernel_size=2, stride=2
        )

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f * 2,
                out_channels=self.f,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.f,
                out_channels=self.f,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(
                in_channels=self.f, out_channels=1, kernel_size=3, padding=1, stride=1
            ),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1
            ),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        # print('input', x.shape)
        x = x.cuda()
        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)  ##池化
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv5', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)
        # print('after conv4', x.shape)

        # Midpoint
        x = self.conv5_block(x)
        # print('mid', x.shape)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)
        # print('final', x.shape)

        return x
