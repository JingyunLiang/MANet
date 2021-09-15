import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from collections import OrderedDict
import models.modules.module_util as mutil


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# MAConv and MABlock for MANet
# --------------------------------------------

class MAConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                          kernel_size=1, stride=1, padding=0, bias=True),
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, 
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)


class MABlock(nn.Module):
    ''' Residual block based on MAConv '''
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super(MABlock, self).__init__()

        self.res = nn.Sequential(*[
            MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction),
            nn.ReLU(inplace=True),
            MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction),
        ])

    def forward(self, x):
        return x + self.res(x)


# -------------------------------------------------
# SFT layer and RRDB block for non-blind RRDB-SFT
# -------------------------------------------------

class SFT_Layer(nn.Module):
    ''' SFT layer '''
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class ResidualDenseBlock_5C(nn.Module):
    '''  Residual Dense Block '''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_SFT(nn.Module):
    ''' Residual in Residual Dense Block with SFT layer '''

    def __init__(self, nf, gc=32, para=15):
        super(RRDB_SFT, self).__init__()
        self.SFT = SFT_Layer(nf=nf, para=para)
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, input):
        out = self.SFT(input[0], input[1])
        out = self.RDB1(out)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return [out * 0.2 + input[0], input[1]]


# ------------------------------------------------------
# MANet and its combinations with non-blind SR
# ------------------------------------------------------

class MANet(nn.Module):
    ''' Network of MANet'''
    def __init__(self, in_nc=3, kernel_size=21, nc=[128, 256], nb=1, split=2):
        super(MANet, self).__init__()
        self.kernel_size = kernel_size

        self.m_head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, padding=1, bias=True)
        self.m_down1 = sequential(*[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)],
                                  nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=2, stride=2, padding=0,
                                            bias=True))

        self.m_body = sequential(*[MABlock(nc[1], nc[1], bias=True, split=split) for _ in range(nb)])

        self.m_up1 = sequential(nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[0],
                                                   kernel_size=2, stride=2, padding=0, bias=True),
                                *[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)])
        self.m_tail = nn.Conv2d(in_channels=nc[0], out_channels=kernel_size ** 2, kernel_size=3, padding=1, bias=True)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x = self.m_body(x2)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        x = self.softmax(x)

        return x


class MANet_s1(nn.Module):
    ''' stage1, train MANet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth',
                 code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2):
        super(MANet_s1, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size

        self.kernel_estimation = MANet(in_nc=in_nc, kernel_size=kernel_size, nc=[manet_nf, manet_nf * 2],
                                       nb=manet_nb, split=split)

    def forward(self, x, gt_K):
        # kernel estimation
        kernel = self.kernel_estimation(x)
        kernel = F.interpolate(kernel, scale_factor=self.scale, mode='nearest').flatten(2).permute(0, 2, 1)
        kernel = kernel.view(-1, kernel.size(1), self.kernel_size, self.kernel_size)

        # no meaning
        with torch.no_grad():
            out = F.interpolate(x, scale_factor=self.scale, mode='nearest')

        return out, kernel


class MANet_s2(nn.Module):
    ''' stage2, train nonblind RRDB-SFT'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth',
                 code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2):
        super(MANet_s2, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size

        self.register_buffer('pca_matrix', torch.load(pca_path).unsqueeze(0).unsqueeze(3).unsqueeze(4))
        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=code_length)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.PixelShuffle(scale))

    def forward(self, x, gt_K):
        # GT kernel preprocessing
        with torch.no_grad():
            kernel_pca_code = torch.mm(gt_K.flatten(1), self.pca_matrix.squeeze()) \
                .unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            # no meaning
            kernel = gt_K

        # nonblind sr
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, kernel_pca_code])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)

        return out, kernel


class MANet_s3(nn.Module):
    ''' stage3, fine-tune nonblind SR model based on MANet predictions'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth',
                 code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2):
        super(MANet_s3, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size

        self.kernel_estimation = MANet(in_nc=in_nc, kernel_size=kernel_size, nc=[manet_nf, manet_nf * 2],
                                       nb=manet_nb, split=split)

        self.register_buffer('pca_matrix', torch.load(pca_path).unsqueeze(0).unsqueeze(3).unsqueeze(4))
        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=code_length)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.PixelShuffle(scale))

    def forward(self, x, gt_K):
        # kernel estimation
        with torch.no_grad():
            kernel = self.kernel_estimation(x)
            kernel_pca_code = (kernel.unsqueeze(2) * self.pca_matrix).sum(1, keepdim=False)
            kernel = F.interpolate(kernel, scale_factor=self.scale, mode='nearest').flatten(2).permute(0, 2, 1)
            kernel = kernel.view(-1, kernel.size(1), self.kernel_size, self.kernel_size)

        # nonblind sr
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, kernel_pca_code])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)

        return out, kernel


if __name__ == '__main__':
    model = MANet_s3()
    print(model)

    x = torch.randn((2, 3, 100, 100))
    x, k = model(x, 0)
    print(x.shape, k.shape)
