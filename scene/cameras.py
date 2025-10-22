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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

# Camera类：把 COLMAP（或 Blender）导出的相机信息、图像、深度图等内容封装成可供 GPU 计算的结构。
class Camera(nn.Module):
    """
    - resolution：图像分辨率
    - FoVx, FoVy：水平、垂直视场角（Field of View）
    - depth_params：深度图的缩放、偏移参数
    - trans, scale：来自scene_info.nerf_normalization，用于场景归一化/坐标变换。trans：三维平移向量。scale：缩放因子。
    - data_device：数据加载到的GPU设备名
    """
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__() # 保证torch.nn.Module父类的初始化

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try: # 尝试使用用户指定的设备
            self.data_device = torch.device(data_device)
        except Exception as e: # 出错则回溯到默认的cuda
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution) # 把PIL图像转成形状为 [C, H, W] 的张量，同时调整到给定分辨率。C为通道数，即RGB或RGBA。
        gt_image = resized_image_rgb[:3, ...] # 将张量的前三个通道（RGB）切片，存储到gt_image中，用来在训练中与渲染图像作比较。
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4: # 若有第四个通道A
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device) # 切片并移动到GPU上
        else: 
            # 若没有α通道，则令所有像素对应α掩码均为1
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
        
        # 曝光分割实验：对每张图像的一部分用于训练，一部分用于测试曝光一致性
        # 把图像左半/右半的 alpha_mask 置零，实现一半区域不参与训练
        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
        
        # clamp（0，1）：min(max(x,0.0),1.0)，生成新的张量，强制所有RGB值均落在0与1之间。
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 可选深度监督实现
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            # 默认所有像素掩码均为1。此外，depth_mask也会复制alpha_mask所在的device。
            self.depth_mask = torch.ones_like(self.alpha_mask) 
            # 调整分辨率。cv2.resize：将invdepthmap缩放到resolution大小。
            self.invdepthmap = cv2.resize(invdepthmap, resolution) 
            # 过滤“负深度值”
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            # depth_params的检查与尺度校正
            if depth_params is not None:
                # 若scake与mws_scale差距过大，则认为深度异常。
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    # 对invdepthmap作线性变换：inv' = inv * scale + offset
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                # 选择第一通道作为有效深度
                self.invdepthmap = self.invdepthmap[..., 0]
            # 转为 torch 张量并移动到 device
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0 # 设置远平面参数（用于投影矩阵）
        self.znear = 0.01 # 设置近平面参数（用于投影矩阵）

        self.trans = trans
        self.scale = scale

        # 用getWorld2View2构造一个从齐次世界坐标到相机坐标的4×4变换矩阵，转置后转化为pytorch张量。
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 用getProjectionMatrix构造4×4的投影矩阵，用来把相机坐标映射到裁剪空间。转置后转化为pytorch向量。
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # 将上面View和Projection两个矩阵结合起来，可以直接把世界齐次坐标向量变换到裁剪空间中。
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 从world_view_transform中提取相机光心（camera center）在世界坐标系中的坐标。
        # 由于V*C_world ​= [0,0,0,1]T，所以C_world ​= V^(−1)*[0,0,0,1]T。但由于前面将V转置，因此C_world_1 = [0,0,0,1]*V_transpose^(-1)。
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

