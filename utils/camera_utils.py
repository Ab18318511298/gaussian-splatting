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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

# 读取单个摄像机图片的数据，构建并返回一个Camera对象。
def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # 使用PIL打开RGB图片
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "": # 如果存在深度图路径
        try:
            if is_nerf_synthetic:
                # 用OpenCV读取深度图，-1表示读取完整图片，包括α通道。读入后转换为 np.float32并做缩放。
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                # / float(2**16)用来把16-bit的深度正则化到[0, 1]范围。
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        # 捕捉三种异常。raise意为“重新抛出”，避免静默失败。
        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    # 接下来计算输出的分辨率
    orig_w, orig_h = image.size # 得到原始宽高
    if args.resolution in [1, 2, 4, 8]: # 如果为离散因子[1, 2, 4, 8]之一
        # 把resolution_scale * args.resolution作为缩放比例，将图片原始宽高压缩成“新的输出分辨率”。
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))

    else:  # 如果是其他值（-1或者一个具体的图像的像素宽度，如800像素）
        if args.resolution == -1: # -1代表“自动模式”，存在阈值1600。
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down) * float(resolution_scale) # 两个缩放因子的乘积
        resolution = (int(orig_w / scale), int(orig_h / scale)) # 除以缩放因子并取整后，得到最终的输出分辨率

    # 将所有数据和信息打包成一个Camera对象
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

# 封装了上面的loadCam()函数。在scene的_init_.py文件中，用于批量构建相机信息列表（训练集、测试集）。
def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    # c为cam_infos中的一个CameraInfo对象。
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    # 输出一个包含所有相机对象的 Python 列表
    return camera_list

# 把Camera对象转换为可写入.json文件的Python字典格式
def camera_to_JSON(id, camera : Camera):
    # Rt存储相机到世界的变换矩阵
    Rt = np.zeros((4, 4))
    # camera.R表示世界到相机，因此要transpose()
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0 # Rt=[[R_wc​_3×3, T_3×1], [0_1×3, 1]​]

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    # numpy数组无法存入json文件，需要用tolist()，将3×3矩阵转换为python的嵌套list
    serializable_array_2d = [x.tolist() for x in rot]
    # 构造最终写入JSON文件的python字典。
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
