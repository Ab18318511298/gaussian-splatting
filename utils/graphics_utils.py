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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# 该函数使用4×4的齐次变换矩阵transf_matrix，对3D点坐标points进行几何变换。
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    # 构造齐次坐标，将每个点从[x, y, z]变为[x, y, z, 1]，使得points_hom形状为（P，4）。
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    # 对(P, 4)的tensor与增维后的(1, 4, 4)tensor进行matmul()运算，得到的结果是(1, P, 4)，其中1是batch维度。
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    # points_out[..., 3:]中的...为“匹配前面所有维度”，因此形状为(1, P, 1)。
    denom = points_out[..., 3:] + 0.0000001
    # 用形状为(1, P, 3)的坐标除以缩放因子denom，然后用.squeeze(dim=0)去除多余的第一维，得到最终的返回结果：3D坐标张量，形状为(P, 3)。
    return (points_out[..., :3] / denom).squeeze(dim=0)

# 通过旋转矩阵、平移向量，返回一个世界到相机的4×4平移变换矩阵。
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# 对上一函数的扩展，仍然返回世界到相机的4×4齐次矩阵，但相机中心的位置经过了平移+缩放
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    # 对W2C求逆得到C2W
    C2W = np.linalg.inv(Rt) 
    # C2W中的平移向量为相机中心在世界坐标系中的位置。
    cam_center = C2W[:3, 3] 
    # 对相机中心的位置进行平移+按比例缩放（归一化）
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
viewpoint
# 构造透视投影矩阵，把齐次化3D点（[x, y, z, 1]）从相机坐标系投影到裁剪空间（二维平面）上，得到[x', y', z', w']。
# 为了从裁剪空间到像素平面，再做齐次除法，映射到NDC标准化坐标系（[x'/w', y'/w', z'/w']），再通过视口变换映射到屏幕像素坐标。
def getProjectionMatrix(znear, zfar, fovX, fovY):
    # tanHalfFov表示屏幕半宽与深度的比例。
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # 定义近平面上视锥体的四条边界
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    
    # 图形系统采用“右手系”则取1；“左手系”取-1。
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# 该函数用视场角fov、图像分辨率（宽与高）求焦距focal
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
    
# 该函数用焦距focal、图像分辨率（宽与高）求视场角fov
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
