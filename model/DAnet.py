import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, factor, angRes):
        super(Net, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.factor = factor
        self.gen_code = Gen_Code(15)
        self.initial_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.deep_conv = CascadeGroups(n_group, n_block, angRes, channels)
        self.up_sample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(factor),
            nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, data):
        (lf, blur, noise) = data
        code = torch.cat((self.gen_code(blur), noise), dim=1)
        b, u, v, c, h, w = lf.shape
        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        buffer = self.initial_conv(x)
        buffer = rearrange(buffer, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        buffer = self.deep_conv(buffer, code)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        out = self.up_sample(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class CascadeGroups(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeGroups, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(BasicGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer, code)

        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        out = self.conv(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class BasicGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(BasicGroup, self).__init__()
        self.DAB = DABlock(channels)
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DistgBlock(angRes, channels))
        self.block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        buffer = self.DAB(x, code)
        for i in range(self.n_block):
            buffer = self.block[i](buffer)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        out = self.conv(buffer)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class DistgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DistgBlock, self).__init__()
        self.spa_conv = SpaConv(channels, channels)
        self.ang_conv = AngConv(angRes, channels, channels // 4)
        self.epi_conv = EpiConv(angRes, channels, channels // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels + channels // 4, channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        fea_spa = self.spa_conv(x)
        fea_ang = self.ang_conv(x)
        fea_epih = self.epi_conv(x)
        xT = rearrange(x, 'b u v c h w -> b v u c w h')
        fea_epiv = rearrange(self.epi_conv(xT), 'b v u c w h -> b u v c h w')
        fea = torch.cat((fea_spa, fea_ang, fea_epih, fea_epiv), dim=3)
        fea = rearrange(fea, 'b u v c h w -> (b u v) c h w')
        out = self.fuse(fea)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class SpaConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(SpaConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, True))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input = rearrange(x, 'b u v c h w -> (b u v) c h w')
        out = self.conv(input)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class AngConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super(AngConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes * angRes * channel_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(angRes))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input_ang = rearrange(x, 'b u v c h w -> (b h w) c u v')
        out = self.conv(input_ang)
        out = rearrange(out, '(b h w) c u v -> b u v c h w', b=b, h=h, w=w)

        return out


class EpiConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super(EpiConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=1, padding=(0, angRes//2), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes * channel_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            PixelShuffle1D(angRes))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input_epi = rearrange(x, 'b u v c h w -> (b u h) c v w')
        out = self.conv(input_epi)
        out = rearrange(out, '(b u h) c v w -> b u v c h w', b=b, u=u, h=h)

        return out


class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor

        return x.view(b, c, h * self.factor, w)


class DABlock(nn.Module):
    def __init__(self, channels):
        super(DABlock, self).__init__()
        self.generate_kernel = nn.Sequential(
            nn.Conv2d(16, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64 * 9, 1, 1, 0, bias=False))
        self.conv_1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.ca_layer = CA_Layer(80, 64)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, code_array):
        b, u, v, c, h, w = x.shape
        input_spa = rearrange(x, 'b u v c h w -> 1 (b u v c) h w', b=b, u=u, v=v)
        kernel = self.generate_kernel(code_array)  # b, 64 * 9, u, v
        kernel = rearrange(kernel, 'b c u v -> (b u v) c')
        fea_spa = self.relu(F.conv2d(input_spa, kernel.view(-1, 1, 3, 3), groups=b * u * v * c, padding=1))
        fea_spa = rearrange(fea_spa, '1 (b u v c) h w -> (b u v) c h w', b=b, u=u, v=v, c=c)
        fea_spa_da = self.conv_1x1(fea_spa)
        fea_spa_da = rearrange(fea_spa_da, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        out = fea_spa_da + self.ca_layer(fea_spa_da, code_array) + x

        return out


class CA_Layer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(CA_Layer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channel_in, 16, 1, 1, 0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, channel_out, 1, 1, 0),
            nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, code):
        b, u, v, c, h, w = x.shape
        fea = rearrange(x, 'b u v c h w -> (b u v) c h w')
        code_fea = self.avg_pool(fea)
        code_deg = rearrange(code, 'b c u v -> (b u v) c 1 1')
        code = torch.cat((code_fea, code_deg), dim=1)
        att = self.mlp(code)
        att = att.repeat(1, 1, h, w)
        out = fea * att
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class Gen_Code(nn.Module):
    def __init__(self, channel_out):
        super(Gen_Code, self).__init__()
        kernel_size = 21
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
        yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
        self.xx_yy = -(xx ** 2 + yy ** 2)
        self.gen_code = nn.Sequential(
            nn.Conv2d(kernel_size ** 2, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, channel_out, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, sigma):
        b, c, u, v = sigma.shape

        kernel = torch.exp(self.xx_yy.to(sigma.device) / (2. * sigma.view(-1, 1, 1) ** 2))
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        kernel = rearrange(kernel, '(b u v) h w -> b (h w) u v', b=b, u=u, v=v)
        code = self.gen_code(kernel)

        return code


if __name__ == "__main__":
    angRes = 5
    factor = 4
    net = Net(factor, angRes)

    from thop import profile
    input_lf = torch.randn(4, angRes, angRes, 3, 32, 32)
    blur = torch.randn(4, 1, angRes, angRes)
    noise = torch.randn(4, 1, angRes, angRes)
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=((input_lf, blur, noise), ))

    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))


