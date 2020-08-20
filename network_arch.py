import torch
import torch.nn as nn
import torch.nn.init as init

"""
Neural Network Architecture of Deep Decomposition Network (DDN) for inverse problems in the paper
@inproceedings{chen2020decomposition,
	author = {Chen, Dongdong and Davies, Mike E},
	title = {Deep Decomposition Learning for Inverse Imaging Problems},
	booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2020}
}
"""

class DDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, operator=None,
                 F='tiny_dncnn', G='compact_unet', connection_type='cascade'):
        super(DDN, self).__init__()
	
        F_arch_hub = {'dncnn':DnCNNBlock, 
                      'tiny_dncnn':TinyDenoiser}

        G_arch_hub = {'unet':UNet, 
                      'compact_unet':CompactUNet}

        assert F in F_arch_hub.keys(), 'Only dncnn, tiny_dncnn are tested!'
        assert G in G_arch_hub.keys(), 'Only unet, compact_unet are tested!'

        self.F = F_arch_hub[F](in_channels=in_channels, out_channels=out_channels)
        self.G = G_arch_hub[G](in_channels=in_channels, out_channels=out_channels)

        self.operator = operator
        self.connection_type= connection_type

    def forward(self, HTy, Pr, Pn):
        assert Pr is not None, 'The physic model/operator Pr=H^+H is required!'
        assert Pn is not None, 'The physic model/operator Pn=I-Pr is required!'

        F = self.F(HTy)
        f = Pr(F)
        if self.connection_type=='cascade':
            g = Pn(self.G(HTy - f))
        if self.connection_type=='independent':
            g = Pn(self.G(HTy))

        return HTy - f + g, F, f, g


class TinyDenoiser(torch.nn.Module):
    def __init__(self,in_channels=10, out_channels=10, depth=4, n_feas=64):
        super(TinyDenoiser, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=n_feas, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(torch.nn.Conv2d(in_channels=n_feas, out_channels=n_feas, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(torch.nn.BatchNorm2d(n_feas, eps=0.0001, momentum = 0.95))
            # layers.append(torch.nn.InstanceNorm2d(n_feas, affine=True))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(in_channels=n_feas, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = torch.nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.dncnn(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class CompactUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(CompactUNet, self).__init__()
        self.name = 'unet'
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=cat_dim)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=cat_dim)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding path
        cat_dim = 1
        input = x
        x1 = self.Conv1(input)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=cat_dim)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=cat_dim)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=cat_dim)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=cat_dim)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DnCNNBlock(nn.Module):
    def __init__(self, depth=17, n_channels=64, in_channels=1, out_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNNBlock, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.dncnn(x)
        # y = x
        # out = self.dncnn(x)
        # return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
