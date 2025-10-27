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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # 创建一个与xyz大小相同（[N, 3]）的全零tensor，用来追踪屏幕空间坐标的梯度。
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # 在bakcward()进行反向传播时，只有“叶子张量”会将梯度存储在grad属性中，从而用于梯度下降。但对“非叶子张量”，也可以显式调用retain_grad()来保留梯度的存储。
    # 用try-except捕获异常，保证后续可以获取“每个点的屏幕空间梯度”
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # tan(FoVx / 2)、tan(FoVy / 2)表示屏幕半宽与深度的比例关系，后面用于计算屏幕坐标、像素位置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 建立统一的光栅化配置，然后创建渲染器对象
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform, # 世界到相机变换矩阵（4×4）
        projmatrix=viewpoint_camera.full_proj_transform, # 相机投影到屏幕归一化坐标系（4×4）
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, # 控制是否提前对 splat 进行模糊处理（通常 False）
        debug=pipe.debug,
        antialiasing=pipe.antialiasing # 抗锯齿开关，控制像素边缘平滑处理
    )

    # 实例化渲染器对象，初始化 GPU 光栅化引擎。完成操作：高斯点投影、计算投影半径radii、进行blending、输出图像。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python: # 如果提前在Python中计算好协方差阵Σ，不需要CUDA在渲染时计算
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: # 否则在CUDA上用R、S自动计算Σ。
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python: # 如果在Python端把球谐系数转成颜色
            # 对[N, 3*16]的二维tensor，用view()重新reshape成[N, 3, 16]的结构。
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 用xyz点减去repeat()展开的相机中心，得到每个点的“方向向量”，从相机中心指向高斯点。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 对方向向量进行单位化：除以自身范数。
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 根据阶数、方向、球谐系数，用eval_sh()函数将sh转换为rgb
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 通过+0.5将rgb值平移到正区间，再用clamp_min(..., 0.0)防止出现负值
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else: # 否则把SH系数传入渲染器，由CUDA内核在渲染时计算
            if separate_sh: # 如果需要将0阶与高阶系数分离
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else: # 如果用户提供了固定颜色（多用于调试）
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rasterizer()将高斯点splat到2D屏幕、得到2D高斯协方差、得到颜色（计算或直接取）、α—blending，并输出“rgb图像、投影半径、深度图”。
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
