import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiDegrade(object):
    def __init__(self, scale=4, kernel_size=21, sig_min=0, sig_max=4.0, noise=None, sig=None):
        self.kernel_size = kernel_size
        self.scale = scale
        if sig is not None:
            sig_min, sig_max = sig, sig
        self.gen_kernel = Isotropic_Gaussian_Blur(kernel_size, [sig_min, sig_max])
        self.blur_by_kernel = Blur_by_Kernel(kernel_size=kernel_size)
        self.bicubic = bicubic()

        if noise is not None:
            self.noise, self.noise_rand = noise, 0  # inference stage
        else:
            self.noise, self.noise_rand = 75, 1   # training stage

    def __call__(self, hr):
        [B, N, C, H, W] = hr.size()
        kernels, sigma = self.gen_kernel(hr.size(0))
        kernels, sigma = kernels.to(hr.device), sigma.to(hr.device)
        hr = rearrange(hr, 'b n c h w -> b (n c) h w')
        hr_blured = self.blur_by_kernel(hr, kernels)
        lr_blured = self.bicubic(hr_blured, scale=1/self.scale)
        lr_blured = rearrange(lr_blured, 'b (n c) h w -> b n c h w', n=N, c=C)

        if self.noise > 0:
            B, N, C, H_lr, W_lr = lr_blured.size()
            if self.noise_rand == 1:
                noise_level = torch.rand(B, 1, 1, 1, 1).to(lr_blured.device)
            else:
                noise_level = torch.ones(B, 1, 1, 1, 1).to(lr_blured.device)
            noise_level = noise_level.mul_(self.noise / 75)
            noise = torch.randn_like(lr_blured).mul_(noise_level * 75 / 255)
            lr_blured.add_(noise)
        else:
            noise_level = torch.zeros(B, 1, 1, 1, 1).to(lr_blured.device)

        return lr_blured, sigma, noise_level.squeeze(1)


class bicubic(nn.Module):
    def __init__(self):
        super(bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)
        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)
        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)
        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)
        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)
        P = np.ceil(kernel_width) + 2
        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)
        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)
        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))
        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)
        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]
        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]
        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        b, c, h, w = input.shape
        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0].to(input.device)
        weight1 = weight1[0].to(input.device)
        indice0 = indice0[0].long()
        indice1 = indice1[0].long()
        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.sum(out, dim=3)
        A = out.permute(0, 1, 3, 2)
        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out


class Blur_by_Kernel(nn.Module):
    def __init__(self, kernel_size=21):
        super(Blur_by_Kernel, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.padding = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.padding = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        [B, C, H, W] = input.size()
        input_pad = self.padding(input)
        output = F.conv2d(input_pad.transpose(0, 1), kernel.unsqueeze(1), groups=B)
        output = output.transpose(0, 1)
        return output


class Isotropic_Gaussian_Blur(object):
    def __init__(self, kernel_size, sig_range):
        self.kernel_size = kernel_size
        self.sig_min = sig_range[0]
        self.sig_max = sig_range[1]
        ax = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        xx = ax.repeat(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        yy = ax.repeat_interleave(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        self.xx_yy = -(xx ** 2 + yy ** 2)

    def __call__(self, batch, sig=None):
        sig_min, sig_max = self.sig_min, self.sig_max
        sigma = torch.rand(batch) * (sig_max - sig_min) + sig_min + 1e-6

        kernel = torch.exp(self.xx_yy / (2. * sigma.view(-1, 1, 1) ** 2))
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        return kernel, sigma


if __name__ == "__main__":
    gen_LR = MultiDegrade()
    angRes = 5
    input = torch.randn(4, 25, 3, 128, 128).cuda()
    output = gen_LR(input)
