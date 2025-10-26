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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    # 指定高斯点各参数的激活函数，以及如何将R、s转化为协方差矩阵Σ
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # build_scaling_rotation()返回矩阵R @ S，是从标准高斯（单位球）到椭圆高斯的线性变换，
            # 其中R为四元数rotation经过qvec2rotmat()变换成的旋转矩阵，S为diag(scaling)。R与S均为[N, 3, 3]。
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 协方差阵Σ = L * L_transpose。transpose(1, 2)表示交换第二、第三维度
            actual_covariance = L @ L.transpose(1, 2)
            # strip_symmetric()会执行(M + M.transpose(1, 2)) * 0.5，完全消除“不对称”的极小误差。
            symm = strip_symmetric(actual_covariance)
            return symm

        # 缩放参数使用exp激活函数保证必为正数。
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # 协方差阵直接由上面的函数定义
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 由于不透明度α取[0, 1]，因此使用sigmoid激活函数
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        
        #对旋转向量单位化，保证只旋转不缩放
        self.rotation_activation = torch.nn.functional.normalize

    # 初始化一个“空”的高斯点云模型对象，定义所有关键参数的容器（Tensor）
    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0 # 当前的球谐函数阶数
        self.optimizer_type = optimizer_type # 优化器类型
        self.max_sh_degree = sh_degree # 最高球谐阶数
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0) # 0阶球谐特征
        self._features_rest = torch.empty(0) # 高阶球谐特征
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0) # 每个点在图像空间的最大半径
        self.xyz_gradient_accum = torch.empty(0) # 累计每个点的梯度模平方
        self.denom = torch.empty(0) # 记录每个点更新的次数
        self.optimizer = None # 优化器对象
        self.percent_dense = 0 # 稠密化比例，在density过程中用来筛选尺度大小
        self.spatial_lr_scale = 0 # 空间学习率缩放
        self.setup_functions()

    # 保存模型的当前参数状态和优化器状态，打包成一个不可变的元组（tuple），和restore()相对。
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    # 从保存的模型快照（tuple）中恢复出完整的高斯点云模型状态。
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args # model_args来自capture()保存的tuple。
        self.training_setup(training_args) # 根据 training_args 创建一个新的优化器对象（如Adam）
        # 恢复梯度统计量
        self.xyz_gradient_accum = xyz_gradient_accum 
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property # 提供访问属性接口，使得可以像成员变量一样，用“调用属性”的方法来调用该函数（如model.get_scaling、model.get_rotation）
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz # 返回所有高斯点的三维坐标[N, 3]
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) # 返回拼接成的[N, 3 * 16]特征矩阵
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure
    
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None: # 若设置为“使用训练阶段学习到的曝光参数”
            # self.exposure_mapping 是一个字典：{image_name: exposure_index}
            return self._exposure[self.exposure_mapping[image_name]]
        else: # 若设置为“使用预训练或外部曝光参数”
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self): # 用来提升当前使用的球谐函数阶数
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 该函数用来从点云初始化整个 3D Gaussian 模型的参数。
    # pcd：输入的点云数据（含坐标和颜色）。cam_infos：所有训练图像的相机信息。spatial_lr_scale：空间学习率缩放因子
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # 将原始点云坐标转为tensor，放进GPU。
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # 将原始RGB颜色转为tensor后，用RGB2SH()转为球谐0阶系数。
        # 构建存储球谐系数的特征张量features，大小为[num_points, 3, 16]。
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0]) # 输出确认初始化规模

        """
        计算每个点的初始尺度。
        distCUDA2()：用来计算与最近k个点的平均欧式距离平方（论文中k=3）。
        clamp_min()：只要tensor中有数据小于给定的下限0.0000001，就直接赋值下限，避免重叠点的距离计算为0。
        """
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 将dist2取平方根（由于距离平方）后，取log（由于激活函数为exp），再扩展到三个完整的维度，得到“各向同性”的球形高斯点的初始化尺度scales
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 初始化N个单位四元数(1, 0, 0, 0)，表示无旋转
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 设定初始透明度均为0.1，然后调用inverse_opacity_activation()，用sigmoid的反函数logit存储初始透明度。
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将这些属性注册为“可梯度优化的参数”
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 初始化辅助张量，用于在投影时记录每个点在屏幕上的最大半径
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 创建exposure_mapping字典，记录image_name → exposure_index，供 get_exposure_from_name() 使用。
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        # eye()用来创立2D的仿射矩阵。eye(3, 4)：tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])。
        # repeat(len(cam_infos), 1, 1)对每个图像都初始化一个仿射矩阵。
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    # 该函数在训练时，决定各个可学习参数（点的位置、特征、旋转、缩放、透明度等）的学习率与调度策略
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        # 初始化存储每个点的“累积梯度模平方”，即在该点的梯度向量(∇x, ∇y, ∇z)的平方和，表示点“位置的梯度强度”。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 初始化存储每个点的更新次数
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 为参数分组，每个参数组都设置字典，并设置不同的学习率策略。
        # l：包含多个字典的列表。lr：该参数组的学习率。name：参数组名字。
        l = [
            # 由于位置xyz变化幅度大，因此采用“较高初始学习率 × 空间缩放系数”的策略
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # 由于球谐高阶特征变化平缓，因此学习率策略要比0阶球谐系数小很多。
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 优化器选择
        if self.optimizer_type == "default": # 默认使用pytorch的Adam优化器
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) # 这里的lr=0.0仅当上面参数组“未定义学习率策略”时采用。
        elif self.optimizer_type == "sparse_adam":
            try: # SparseGaussianAdam 是为 3DGS 优化设计的版本，它可以高效地在稀疏点集上只更新可见的高斯。
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 单独为[N, 3, 4]曝光矩阵设置好优化器Adam，因为曝光矩阵的更新规律、学习率、调度策略都与高斯参数（位置、缩放、颜色等）不同。
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # get_expon_lr_func()会返回一个“指数衰减调度函数”，让位置xyz初期快速收敛，后期逐步减小步长以优化细节。
        # lr_delay_steps若未设置，则默认为0，即没有“warm-up”阶段，直接进行指数衰减调度。
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        # 为曝光矩阵设置独立的指数衰减调度，因为曝光参数往往收敛较慢，需要不同的调度周期。
        # 对exposure的优化，设置了lr_delay_steps，说明在前期存在“warm-up”阶段，学习率从lr_delay_mult * lr_init开始，上升到lr_mult。避免训练初期剧烈震荡。
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    # 该函数根据当前的迭代步数更新不同参数组的学习率。
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None: # 如果没有预训练的曝光参数
            # param_groups是一个长度为1的列表，元素是整个曝光张量[N, 3, 4]。
            # 循环只会进行一次，更新这个“包含不同图像”的参数组的学习率。
            for param_group in self.exposure_optimizer.param_groups:
                # 使用带warm-up的指数衰减调度函数exposure_scheduler_args()来更新。
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        # 遍历字典中的参数组
        for param_group in self.optimizer.param_groups:
            # 只更新位置xyz的学习率。
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # 该函数生成一个字符串列表，每个元素对应高斯点的一个属性名称
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        # _features_dc：[num_points, 3, 1]；_features_rest：[num_points, 3, 15]
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i)) # 命名f_dc_0 、 f_dc_1 、 f_dc_2（3个属性）
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i)) # 命名f_rest_0 、 f_rest_1 、 ...... 、 f_rest_44（45个属性）
        
        l.append('opacity') # 添加不透明度
        
        # scaling：[N, 3]；rotation：[N, 4]
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # 把当前的高斯点模型参数（位置、特征、尺度、旋转、不透明度等）保存为 .ply 文件（Point Cloud 格式）
    def save_ply(self, path):
        # mkdir_p等价于 os.makedirs(..., exist_ok=True)，确保文件路径存在，否则创建路径。
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # 无法线信息，只创建与xyz同大小的0矩阵[N, 3]
        normals = np.zeros_like(xyz) 
        # transpose(1, 2).flatten(start_dim=1)：将[N, 3, 1]变为[N, 1, 3]，再变为[N, 3]（为了与平铺的属性名对应）。contiguous() 确保内存连续
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # transpose(1, 2).flatten(start_dim=1)：将[N, 3, 15]变为[N, 15, 3]，再变为[N, 45]（为了与平铺的属性名对应）。contiguous() 确保内存连续
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 构造.ply文件的数据结构
        # 'f4'：每个属性名对应一个float32数据。construct_list_of_attributes()得到属性名列表。
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full) # 每个高斯点是 .ply 文件中的一个顶点
        # concatenate()将所有属性拼接在一起，拼成二维矩阵[N, total_dim]。
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # map(tuple, attributes)将每一行（单个高斯点的所有属性）转为一个元组，将元组属性分别横向填入elements这个结构化数组。
        # elements给每一列都起了名字、数据类型，共有total_dim列；行则为索引，每一行表示一个高斯点，共有N行。
        elements[:] = list(map(tuple, attributes))
        # 写入.ply文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 该函数“正则化”重置不透明度α，以控制高斯数量的过度增长
    def reset_opacity(self):
        # 将经过激活函数后的get_opacity所有值设置上限0.01，即“重置”回接近透明的状态。再用激活函数的反函数，得到新的可优化参数的tensor。
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # replace_tensor_to_optimizer()：将新的张量替换原优化器中对应的参数"opacity"。输出一个字典 optimizable_tensors，包含更新后的张量。
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"] # 更新模型参数_opacity为新tensor。

    # 把保存在.ply文件中的高斯点云特征重新加载回GaussianModel的可训练参数中。
    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            # 如果存在expose.json文件
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    # 读入所有图像的曝光矩阵
                    exposures = json.load(f)
                    # 存储在“预训练曝光参数”中。
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        # 恢复每个点的xyz（[N, 3]）和opacity（[N, 1]）
        # np.asarray()同np.array()的区别在于，在数据已经是numpy数组时不会进行复制。
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # [..., np.newaxis]用于增加新维度，将一维数组[N]变为二维数组[N, 1]。
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 如果列的属性名以f_rest_开头，则取出来放入extra_f_names
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # sorted()：对列表进行排序并返回新的列表。lambda：用来定义匿名函数，等价于def key(x): return x。x.split('_')[-1]：把x按下划线'_'分割，取最后一段（如"f_rest_0"取0，便于排序）
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # 检查名字数量是否和高阶球谐系数相等（45个）
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # 创建二位numpy数组[N, 45]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将 numpy 数据转到 GPU 上，声明为 nn.Parameter，从而变回可训练参数。
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 该函数能够将优化器中某一个参数tensor替换为新tensor，用于“正则化”重置不透明度α时，用新的opacity张量替换优化器中的参数“opacity”。
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 当参数字典的“name”字段等于输入的name
            if group["name"] == name: 
                '''
                通过旧参数找到存储的动量状态。
                state是一个字典，键是参数的标识符，字段是另一个字典，包括：
                - exp_avg：一阶动量，通过对历史一阶动量与当前梯度进行加权平均得出，即历史梯度的指数加权平均
                - exp_avg_sq；二阶动量，通过对历史二阶动量与当前梯度平方进行加权平均得出，即历史梯度平方的指数平均
                - step：迭代更新次数
                '''
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 将动量状态重置为0张量
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 删除state中旧参数的状态信息
                del self.optimizer.state[group['params'][0]]
                # nn.Parameter()将输入的新tensor注册为可优化参数，替换参数组中的旧tensor
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                # 将重置后的动量状态绑定到新tensor上
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 根据布尔掩码 mask 删除（裁剪）一部分高斯点，即同时删除其对应的所有参数属性。
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None: # 如果优化器有保存该参数的动量状态（即它有被训练过），则必须同时修改参数组和优化器中对应的状态
                # 对一阶、二阶动量都应用掩码处理，防止与参数组中N大小不一致
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除旧的参数状态
                del self.optimizer.state[group['params'][0]]
                # 对参数张量也应用掩码进行裁剪，重新注册为可优化参数
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else: # 若没有动量状态，说明未训练过，只需修改参数组字典，无需修改优化器
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # mask=True说明该点需要删除，~取反后符合删除逻辑
        valid_points_mask = ~mask
        # 应用_prune_optimizer()对优化器中所有参数执行剪枝
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 把_prune_optimizer()裁剪后的参数更新回模型，此时模型中只包含应该保留的高斯点。
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 同步裁剪缓存的辅助变量
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    # 与_prune_optimizer相对，用来新增一部分高斯点。将一批新张量tensors_dict拼接进参数组字典和优化器中。
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 保证循环中一个组只有一个参数，否则报错
            assert len(group["params"]) == 1
            # 取出要拼接的新张量（如对xyz坐标新增的[N_new, 3]）
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None: # 如果有状态，则扩展状态tensor
                # 拼接和新增tensor同维度的向量，且由于还未训练，动量全初始化为0
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else: # 如果状态全为空，则只需拼接参数。
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 用于“增密化的后处理”
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        # 把新的高斯点参数打包成一个字典
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 将输入的新参数字典用cat_tensors_to_optimizer()拼接进参数组和优化器中
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # 将优化器中拼接的结果写回模型，让模型真正拥有了这些新点及对应参数。
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # 对于高斯点的2D投影半径这一缓存统计量，直接拼接进模型
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        # 对于和点数量有关的辅助统计量，由于有新点增加，需要完全重置为0，在后续训练中重新累积。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 获取初始高斯点数量
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        # grads是[N, 1]的tensor，squeeze()将grads变成一维的[N]大小的tensor。取出部分点/全部点的梯度强度。
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        '''
        - torch.logical_and：对两个 [N] 布尔向量做元素级 AND。（必须同时满足梯度筛选和尺度筛选）
        - torch.max(self.get_scaling, dim=1).values：对[N, 3]的get_scaling的第一维度（即单个点的xyz）取max值，得到[N]大小的tensor。
        '''
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
