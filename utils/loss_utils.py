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

# 该函数是_ssim()的高层封装函数
def ssim(img1, img2, window_size=11, size_average=True):
    # 由于不确定img是否有batch维度，取“倒数第三维”的通道数C，放入channel。
    channel = img1.size(-3)
    # 用create_window()生成形状为[3, 1, 11, 11]的二维高斯卷积核
    window = create_window(window_size, channel)

    if img1.is_cuda: # 如果img在cuda上
        # 把window放在相同的cuda上。
        window = window.cuda(img1.get_device())
    #确保 window 的数据类型（float16/float32）与img1一致
    window = window.type_as(img1)

    # 调用_ssim()函数得到平均SSIM值（标量或一维向量）。
    return _ssim(img1, img2, window, window_size, channel, size_average)

# 该函数为python实现的SSIM前向传播，与CUDA实现的fusedssim等价。
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 计算局部均值：F.conv2d实现卷积操作（对每个像素区域用window卷积，实现局部加权平均）；groups=channel表示每个通道独立卷积
    # 为了保持卷积后输出图像的尺寸不变，使用padding=window_size // 2（在图像的边缘周围填充一些像素），保证输出的map能与图像像素一一对应。
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算局部方差、局部协方差：依旧使用window卷积实现局部加权平均。
    # 方差公式：σ_x ^ 2 = E[x^2] - E[x]^2；协方差公式：σ_xy = E[x*y] - E[x]*E[y]。
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 计算ssim_map（一个tensor），包含每个像素的SSIM值。
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 由于ssim_map与img同大小，为[B, 3, H, W]或[3, H, W]
    if size_average:
        # 对整个batch的所有图像、通道求平均ssim值
        return ssim_map.mean()
    else:
        # 对单个图像的所有通道求平均ssim值。
        return ssim_map.mean(1).mean(1).mean(1)

# 该函数与ssim相对，应用前面的自定义算子FusedSSIMMap，显著加速前向与反向传播。
def fast_ssim(img1, img2):
    # .apply()是由torch.autograd.Function基类“自动定义”的一个静态方法。
    # 使用apply()后，会调用forward()做前向计算、注册计算图节点（保存 ctx、输入输出、以及 backward 的定义，以便反向传播时调用）
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    # 返回整个batch的平均值
    return ssim_map.mean()
