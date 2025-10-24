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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

# 定义了一个“统一标准”的相机信息结构体，让不同来源的数据都转化为该结构。
class CameraInfo(NamedTuple): # 是NamedTuple类型，是个“不可变”的数据容器，能像字典一样通过字段名访问数据
    uid: int
    R: np.array
    T: np.array
    FovY: np.array # 垂直方向视场角
    FovX: np.array # 水平方向视场角
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str # 深度图文件路径（可能为空）
    width: int
    height: int
    is_test: bool

# 从不同数据格式加载场景后返回的标准化接口。
class SceneInfo(NamedTuple): # 同样“不可变”，能通过字段名访问数据
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

# 基于Nerf++的normalization方法：根据相机的外参R、T，计算场景的中心点平移量（translate）与半径（radius），为了把整个场景（点云与相机分布）归一化到标准尺度
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers) # 把所有相机中心拼成一个 3×N 的矩阵，第一行均为x坐标，第二行均为y坐标，第三行均为z坐标
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True) # 计算所有相机中心的均值c_x、c_y、c_z，构成3×1矩阵
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True) # 计算每个相机中心与场景中心的欧氏距离
        diagonal = np.max(dist) # 找到最大距离，即“包围所有相机中心的球体的半径”，也为对角线长度的一半。
        return center.flatten(), diagonal # flatten()将二维矩阵平铺成一维数组(c_x, c_y, c_z)
    
    cam_centers = [] # 创建一个包含多个相机中心的世界坐标（3×1）的列表
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T) # W2C = [[R_3×3, T_3×1], [0_1×3, 1]]
        # 由于x_c​ = R * x_w​ + T，且x_c = 0，则x_w = -T * R_transpose。
        # 又因为C2W = inv(W2C) = [[R_transpose, -R_transpose * T], [0, 1]，因此直接对C2W矩阵作切片可得到相机中心的世界坐标x_w
        C2W = np.linalg.inv(W2C) 
        cam_centers.append(C2W[:3, 3:4]) # 取C2W最后一列前三个元素（3×1向量）

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1 # 略微增大半径，避免相机出现在边缘

    translate = -center # 将相机平均中心点移回世界坐标原点需要的平移量

    return {"translate": translate, "radius": radius}

# 读取并构建每个相机的 CameraInfo 对象（包含内参、外参、图像路径、FOV 等）
def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    # 按照索引号、key（图像名）逐个读取每个图像的相机外参
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # 带进度条，可视化读取过程：
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        # 刷新标准输出流的缓冲区，确保实时显现读取结果
        sys.stdout.flush()

        extr = cam_extrinsics[key] # 依照图像名来提取外参（四元数、平移向量、图像名、对应相机id等）
        intr = cam_intrinsics[extr.camera_id] # 依照对应相机id提取内参（焦距、相机类型、图像宽高等）
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) # 将外参中的四元数转变成3×3旋转矩阵R_C2W，再转置变为R_W2C存储
        T = np.array(extr.tvec) # 转换成numpy数组

        if intr.model=="SIMPLE_PINHOLE": # 单焦距（f）模型
            # 读取内参中的焦距
            focal_length_x = intr.params[0] 
            # focal2fov()用来从焦距计算视场角（单位：弧度）
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE": # 双焦距（fx, fy）模型
            # 读取内参中的两个
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            # focal2fov()用来从焦距计算视场角（单位：弧度）
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # 尝试匹配深度估计参数
        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        # 拼接图像与深度文件路径
        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        # 构建 CameraInfo 类型的对象，每个cam_info都封装一个相机的完整信息
        # is_test布尔值判断image_name是否在test_cam_names_list中。
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos # 返回cam_infos列表

# 从 .ply 点云文件（一种常见的点云/网格格式，可以包含任意属性字段）中读取点的位置、颜色、法向量，并封装成 BasicPointCloud 对象
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# fetchply()的反操作，将内存中的点云数据保存为 .ply 文件
def storePly(path, xyz, rgb):
    # 定义了.ply文件的字段结构：结构化numpy数组
    # f4：float32类型；u1：uint8类型
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz) # 创建一个与 xyz 相同形状的零矩阵，来存储法线

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1) # 把坐标、法线、颜色共9个元素拼接成一行的数组
    elements[:] = list(map(tuple, attributes)) # 把每一行都转为 tuple 并填充到结构化数组中

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# 基于readColmapCameras()，从一个 COLMAP 格式的重建结果目录中读取、解析并标准化场景的所有信息（相机、图像、点云、深度、测试划分、以及归一化参数等），并组装成一个统一的 SceneInfo 对象
def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try: # 优先尝试读取二进制.bin格式，否则退回文本.txt格式
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        # 使用colmap_loader.py中定义的函数来读取相机内外参
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        # 使用colmap_loader.py中定义的函数来读取相机内外参
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 若指定了depths文件夹，则尝试加载深度参数depth_params.json
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            # 以图像名为索引，取出每张图对应的深度缩放比例scale，存储进numpy数组all_scales
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                # 计算出全局中位数med_scale
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                # 把med_scale也补充到每个图像的参数中
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # 划分训练集和测试集。输出结果test_cam_names_list是一个以“测试图像名”为元素的列表
    if eval: # 若eval=True则需要测试图像
        if "360" in path: # 若为LLFF 数据集的 “360 场景”，则每8张图测试一张图
            llffhold = 8
        if llffhold: # 若指定了llffhold，则每达指定数量测试一张图
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else: # 否则直接从 test.txt 文件中读取测试图像名
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else: # 若eval=False则全为训练图像；
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    # 调用readColmapCameras()函数，把 COLMAP 的外参/内参、深度参数、图像目录/深度目录及测试集名单传进去，得到一个未排序的 CameraInfo 列表
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    # 复制一份列表，按照图像名的“字典序”来排序，得到稳定、可重现的相机信息列表cam_infos，该列表是一个list[CameraInfo]
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 基于c.is_test的布尔值标志把 cam_infos 列表拆分为训练集和测试集
    # 当train_test_exp = False（默认值）时区分训练集和测试集。 = True时把所有图像都放进训练集（即不区分训练集和测试集）
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    # 用getNerfppNorm()取得场景平移量、半径，用于归一化场景坐标
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # 准备读取点云，ply格式优先，其次是bin，最后是txt。
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb) # bin和txt读取后需要用storePly()转换为ply格式，返回 BasicPointCloud 对象。
    try:
        pcd = fetchPly(ply_path) # 优先尝试读取.ply点云，返回 BasicPointCloud 对象。
    except:
        pcd = None

    # 将读取的数据均放入scene_info这个SceneInfo类型的标准化容器。
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

# 与readColmapCameras() 作用类似，但从transforms.json（NeRF 数据集的描述文件） 读取
def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"] # 读取水平方向视场角（单位：弧度），这是blender格式唯一的相机内参

        frames = contents["frames"] # frames是列表，每个元素包含“相对图像路径file_path”和4×4的“c2w矩阵transform_matrix”
        for idx, frame in enumerate(frames):
            # 对每个图片，拼接完整的图像路径
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
