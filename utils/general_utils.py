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
import sys
from datetime import datetime
import numpy as np
import random

# 定义sigmoid反函数，用于反激活
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 在scene/cameras.py中，用于把PIL图像转换为torch，并调整为给定分辨率
# pil_image是一个PIL.Igame对象，resolution是目标分辨率元组（width, height）
def PILtoTorch(pil_image, resolution):
    # resize()会新生成一个PIL图像，大小为指定的resolution，默认方法为bilinear（双线性插值）
    resized_image_PIL = pil_image.resize(resolution)
    # 将图片转换成numpy数组，再转为pytorch张量，然后进行归一化（由于数据类型通常为int8，范围为[0, 255]）
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3: # 如果图像为彩色图像[H, W, 3]（即有RGB三通道），则改为pytorch中的标准通道顺序[3, H, W]
        return resized_image.permute(2, 0, 1)
    else: # 如果图片为灰度图像[H, W]，则用.unsqueeze(dim=-1)增维到(H, W, 1)，再修改顺序[1, H, W]
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

# 返回“指数衰减调度函数”，用于xyz等参数的学习率调整。
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    - lr_delay_steps：warm-up预热阶段持续的步数。
    - lr_delay_mult：warm-up阶段的初始倍率，一般为<=1的parameter。学习率会从lr_delay_mult * lr_init开始，逐步提升学习率至lr_init。
    
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    # 定义某一步数下的学习率变化
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0): # 定义无效情况，用于跳过学习率计算
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0: # 如果存在warm-up阶段，则求出预热比率（warm-up步数内小于1，正式衰减时保持1）
            # A kind of reverse cosine decay.
            # 设lr_delay_mult = 0.2，则预热比率delay_rate = 0.2 + 0.8 * sin(步数比 * π/2)，使学习率变化更加平滑。
            # 用np.clip()控制步数比，即使step超出预热阶段，则令步数比恒为1。
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        # 定义全程的步数比。
        t = np.clip(step / max_steps, 0, 1)
        # 对数空间线性插值lr(t) = lr_init ^ (1 − t)​ * lr_final ^ t​，等价于指数衰减公式lr(t) = lr_init * [(lr_final / lr_init) ^ t]。
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        # 返回该步数的最终学习率，初期由delay_rate控制预热，后期delay_rate恒为1。
        return delay_rate * log_lerp

    return helper

# 该函数用来从每个高斯协方差矩阵中，提取出6维向量，便于参数优化，同时保证还原后协方差阵Σ对称。
def strip_lowerdiag(L):
    # L.shape == (N, 3, 3)，是批量的矩阵tensor。创建的uncertainty是大小为[N, 6]的tensor
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    # uncertainty的每一行，保存一个高斯点协方差阵的“上三角六元素”
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# 将旋转四元数“批量”转换成3×3旋转矩阵
def build_rotation(r):
    # 求范数并归一化
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # r为实部w，x、y、z为虚部
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# 构建每个高斯的线性变换矩阵L，Σ = L * L_transpose
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    # s[:,0], s[:,1], s[:,2]对应高斯的三个主轴方向的标准差（或缩放因子）
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# 在正式训练（train.py中）、渲染设置（render.py中）前，都会用该函数初始化环境
# silent = False为“带时间戳的正常打印”，= True为“静默模式，不打印”。
def safe_state(silent):
    # 这段代码劫持标准输出流（即 print() 的目标）。每次打印时都会自动加上时间戳。
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    # 设置随机状态，确保可复现性。
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # 固定GPU设备
    torch.cuda.set_device(torch.device("cuda:0"))
