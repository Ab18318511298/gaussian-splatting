#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
# fusedssim：计算 SSIM map 的前向函数
# fusedssim_backward：计算 SSIM 的梯度（反向传播函数）
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

# SSIM公式中的常数稳定项，这里可防止分母为0。
C1 = 0.01 ** 2
C2 = 0.03 ** 2

# 继承自 torch.autograd.Function，表示“自定义”的前向和反向计算函数
class FusedSSIMMap(torch.autograd.Function):
    # 将forward()与backward()定义为静态方法，...
    # 使得“不创建类实例”时也可以通过FusedSSIMMAP.forward()和FusedSSIMMAP.backward()调用，也无法修改类/实例状态。
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        # ctx是context（上下文对象），用来保存前向传播中的一些信息，以便在反向传播中使用。
        # 使用ctx.save_for_backward()保存反向传播需要的张量，让反向传播可以使用前向传播的中间结果，不需要重复计算。
        # img1.detach()能生成一个新tensor，与原tensor共享数据，但被“切断了梯度跟踪”，不参与梯度计算。
        ctx.save_for_backward(img1.detach(), img2)
        # 保存常量信息。
        ctx.C1 = C1
        ctx.C2 = C2
        # 返回每个像素的SSIM值（通常在[0, 1]之间）
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        '''
        loss.backward()反向传播时，会返回对L1、L_ssim、Depth的梯度，其中对ssim的梯度即为opt_grad（∂loss/∂ssim_map）。
        然后进行FusedSSIMMap()，前向传播得到ssim_map具体数值后，进行backward(ctx, opt_grad)，得到ssim对img1的梯度grad。
        最后autograd会将grad回传到model参数。
        '''
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        # 调用fusedssim_backward()计算梯度，返回loss对img1的梯度张量，形状与img1相同。
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        # 在自定义Function.backward中，返回值的顺序必须严格对应 forward 的输入顺序
        # 即：C1、C2、img2都无需梯度，而img1需要梯度
        return None, None, grad, None

# 定义像素级损失函数L1。
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# 定义“对误差惩罚更强”的像素级损失函数L2。
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

# 该函数生成一个一维高斯核，window_size为生成数的个数。
def gaussian(window_size, sigma):
    # 用2D高斯分布公式创建高斯核：G(x) = e ^ [−2 * σ^2 / (x − μ) ^ 2​]，window_size // 2即中心位置。
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # 返回归一化结果，使所有元素和为1。
    return gauss / gauss.sum()

# 该函数生成一个 二维高斯窗口（卷积核），channel一般为3，表示rgb三通道。常在计算 SSIM 时用作“局部加权平均”。
def create_window(window_size, channel):
    # 函数.unsqueeze(1)将一维高斯权重变为列向量。
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 函数_1D_window.mm(_1D_window.t())计算外积，将n个一维点变成n * n个的二维高斯核。
    # 函数.unsqueeze(0).unsqueeze(0)用来添加 batch 和 channel 维度，变成 [1, 1, H, W]。
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 函数expand(channel, 1, window_size, window_size)将[1, 1, H, W]变成[3, 1, H, W]。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

# 该函数为python实现的SSIM前向传播，与CUDA实现的fusedssim等价。
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 计算局部均值
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
