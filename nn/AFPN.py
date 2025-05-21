from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmyolo.registry import MODELS


def BasicConv(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(inplace=True)),
    ]))


def Conv(filter_in, filter_out, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.silu(out)

        return out


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x

class Upsample1(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample1, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 2, 2)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 4, 4)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x

#2023-10-23  CARAFE
import torch.nn.functional as F
class Upsample(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, scale_factor=2):
        super(Upsample, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = scale_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor
#2023-10-27

class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1,1)

        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ScaleBlockBody(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super(ScaleBlockBody, self).__init__()

        self.blocks_top1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_mid1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_bot1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )

        self.downsample_top1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_mid1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_top1 = ASFF_2(inter_dim=channels[0])
        self.asff_mid1 = ASFF_2(inter_dim=channels[1])

        self.blocks_top2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0])
        )
        self.blocks_mid2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1])
        )

        self.downsample_top2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_top2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_mid2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_mid2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_bot2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_bot2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_top2 = ASFF_3(inter_dim=channels[0])
        self.asff_mid2 = ASFF_3(inter_dim=channels[1])
        self.asff_bot2 = ASFF_3(inter_dim=channels[2])

        self.blocks_top3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0])
        )
        self.blocks_mid3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1])
        )
        self.blocks_bot3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2])
        )

    def forward(self, x):
        x1, x2, x3 = x

        x1 = self.blocks_top1(x1)
        x2 = self.blocks_mid1(x2)
        x3 = self.blocks_bot1(x3)

        top = self.asff_top1(x1, self.upsample_mid1_2(x2))
        mid = self.asff_mid1(self.downsample_top1_2(x1), x2)

        x1 = self.blocks_top2(top)
        x2 = self.blocks_mid2(mid)

        top = self.asff_top2(x1, self.upsample_mid2_2(x2), self.upsample_bot2_4(x3))
        mid = self.asff_mid2(self.downsample_top2_2(x1), x2, self.upsample_bot2_2(x3))
        bot = self.asff_bot2(self.downsample_top2_4(x1), self.downsample_mid2_2(x2), x3)

        top = self.blocks_top3(top)
        mid = self.blocks_mid3(mid)
        bot = self.blocks_bot3(bot)

        return top, mid, bot

@MODELS.register_module()
class YOLOv5AFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[256, 512, 1024]):
        super(YOLOv5AFPN, self).__init__()

        self.conv1 = BasicConv(in_channels[0], in_channels[0] // 4, 1)
        self.conv2 = BasicConv(in_channels[1], in_channels[1] // 4, 1)
        self.conv3 = BasicConv(in_channels[2], in_channels[2] // 4, 1)

        self.body = nn.Sequential(
            ScaleBlockBody([in_channels[0] // 4, in_channels[1] // 4, in_channels[2] // 4])
        )

        self.conv11 = BasicConv(in_channels[0] // 4, out_channels[0], 1)
        self.conv22 = BasicConv(in_channels[1] // 4, out_channels[1], 1)
        self.conv33 = BasicConv(in_channels[2] // 4, out_channels[2], 1)

        # ----------------------------------------------------------------#
        #   init weight
        # ----------------------------------------------------------------#
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x1, x2, x3 = x

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out1, out2, out3 = self.body([x1, x2, x3])

        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        return tuple([out1, out2, out3])