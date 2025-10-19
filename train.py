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

import os
import torch
from random import randint # randint返回指定范围内的一个随机整数
from utils.loss_utils import l1_loss, ssim # 损失函数L1和L_SSIM
from gaussian_renderer import render, network_gui # 核心渲染器和GUI
import sys
from scene import Scene, GaussianModel # 场景管理和高斯模型
from utils.general_utils import safe_state, get_expon_lr_func # 工具函数
import uuid # 用来生成唯一标识符（UUID）
from tqdm import tqdm # 进度条
from utils.image_utils import psnr # PSNR图像质量评估
from argparse import ArgumentParser, Namespace # 命令行参数解析
from arguments import ModelParams, PipelineParams, OptimizationParams # 参数配置类

# 尝试导入SummaryWritter，用于记录训练数据，方便tensorboard进行可视化
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 尝试导入优化后的SSIM（效果与SSIM一样，但计算效率更快）实现
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

# 尝试导入稀疏Adam优化器
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
     """
    - dataset: 数据集配置
    - opt: 包含所有训练优化器相关的超参数，由arguments/__init__.py里的 class OptimizationParams定义。 
    - pipe: 渲染管线参数：在python还是在CUDA转换球谐系数、在python还是在CUDA计算协方差阵、是否启动调试？
    - testing_iterations: 测试迭代点列表
    - saving_iterations: 保存迭代点列表
    - checkpoint_iterations: 检查点迭代点列表
    - checkpoint: 预训练检查点路径
    - debug_from: 从哪个迭代开始调试
    """
    
    # 检查稀疏Adam优化器是否可用
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0 # 起始迭代次数
    tb_writer = prepare_output_and_logger(dataset) # 初始化输出目录和日志记录器，返回一个SummaryWritter实例
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type) # 初始化高斯模型，后者规定优化器类型为'Adam'
    scene = Scene(dataset, gaussians) # 初始化场景，加载数据集和对应的相机参数
    gaussians.training_setup(opt) #根据传入的 opt 参数，设置训练参数（优化器、各种学习率等）。该函数在scene/gaussian_model.py里定义
    
    if checkpoint: # 如果提供了checkpoint，则从checkpoint加载模型参数并恢复训练进度
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] # 设置背景颜色，白色或黑色取决于数据集要求
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 把背景颜色转换为 GPU 上的 float32 张量，以便在渲染或损失计算时作为背景色使用

    # 创建CUDA事件用于对GPU计时
    # 使用方法：iter_start.record()、iter_end.record()、iter_start.elapsed_time(iter_end)
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 是否使用稀疏Adam优化器
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 

    # 创建一个exponential函数，返回深度L1损失的权重。参数有初始权重、最终权重、总训练步数
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy() # 获取所有训练相机对象（包括内参、外参、RGB、深度图等）
    viewpoint_indices = list(range(len(viewpoint_stack))) # 给每个训练相机分配一个索引编号，从0到N-1（假设有N个训练视角）
    ema_loss_for_log = 0.0 # 初始化指数移动平均损失=0，用于日志记录
    ema_Ll1depth_for_log = 0.0 # 初始化指数移动平均深度损失=0

    # 使用tqdm库创建进度条，实时追踪训练进度、耗时、速率。range为训练范围，desc为进度条描述文字。
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    first_iter += 1 # 当first_iter = 5000, 之后要从checkpoint恢复训练，就会从5001次开始，避免重复执行第5000次。

    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):
        
        # 迭代训练前，建立实时的可视化GUI。
        """
        - custom_cam：用户当前在 GUI 中的指定相机视角参数
        - do_training：是否开始/继续训练（GUI按钮控制）
        - pipe.convert_SHs_python：是否在 Python 端处理球谐转换（调试参数）
        - pipe.compute_cov3D_python：是否在 Python 端计算协方差矩阵（调试参数）
        - keep_alive：GUI是否保持连接
        - scaling_modifier：调整高斯点尺寸缩放的参数
        """
        if network_gui.conn == None:
            network_gui.try_connect() # 尝试建立socket通信以连接训练。
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收GUI通过network_gui.receive()发送的消息
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 调用render()渲染用gaussians在用户指定的custom_cam下渲染一帧图像
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    # 将渲染结果转为 8-bit RGB 数组（0–255），再转为字节流，准备发给 GUI。
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                # 把渲染结果（图片）发送回 GUI 客户端显示，即可实时预览画面。
                network_gui.send(net_image_bytes, dataset.source_path)
                # 接下来判断要“开始训练”还是继续“预览画面”。如果按下“开始训练”并且（“迭代还没结束”或不要求连接GUI），就“跳出循环”。
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e: # 异常捕获，防止GUI出问题时妨碍迭代训练。
                network_gui.conn = None
        
        iter_start.record() # GPU计时开始

        gaussians.update_learning_rate(iteration) # 根据当前迭代次数，更新学习率（位置p、SH系数、协方差Σ、不透明度α），位置p使用指数衰减调度

        # 为了避免基色错误，从零阶开始，每1000次迭代，提升SH系数的阶数直到三阶。
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 从“训练集”中随机选择一个视角来训练渲染，并保证epoch内一个视角只出现一次。
        if not viewpoint_stack: # 本轮次视角用完后
            viewpoint_stack = scene.getTrainCameras().copy() # copy一轮新的视角
            viewpoint_indices = list(range(len(viewpoint_stack))) # 同前，创建索引列表
        rand_idx = randint(0, len(viewpoint_indices) - 1) # 从本轮次剩余视角中随机选择一个
        viewpoint_cam = viewpoint_stack.pop(rand_idx) # 从viewpoint_stack取出视角对象后，会删除该元素。
        vind = viewpoint_indices.pop(rand_idx) # 从viewpoint_indices中找到对应索引，然后删除索引号。

        if (iteration - 1) == debug_from: # 如果达到调试起点，则启动调试模式。如果在某阶段出现异常，便不用从头训练排查。
            pipe.debug = True
        
        # 开始前向渲染
        #根据设置决定是“每次随机生成一个RGB值作背景”还是“使用固定的背景张量”。
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        """
        核心渲染语句。
        其中参数use_trained_exp：是否使用训练集曝光参数；separate_sh：是否独立处理球谐系数
        输出结果render_pkg是一个字典：
        {
          "render": image,                  # 渲染得到的RGB图像
          "viewspace_points": tensor,       # 点云在当前视角下的投影坐标
          "visibility_filter": mask,        # 哪些高斯被看到
          "radii": tensor                   # 每个高斯在屏幕上的半径
        }
        """
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # α_mask是个与三通道image同大小的二维矩阵，存储每个像素的α值（0 ~ 1）
        # image.shape = （3，H，W）；alpha_mask.shape = （H，W）
        if viewpoint_cam.alpha_mask is not None: 
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask # image每个通道均与α_mask做“逐元素相乘”，从而使“无效像素”的RGB值清零，达到屏蔽效果。

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() # 获取真实图像
        Ll1 = l1_loss(image, gt_image) #L1损失
        if FUSED_SSIM_AVAILABLE: # SSIM损失（优化版或普通版）
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) # 定义损失函数，其中超参数opt.lambda_dssim被设为0.2

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
