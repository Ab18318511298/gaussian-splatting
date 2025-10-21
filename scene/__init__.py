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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
    
    """
    Scene 类用于管理场景的3D模型，包括相机参数、点云数据和高斯模型的初始化和加载
    """
class Scene:
    
    # 创建一个叫“gaussians”的成员变量，其类型是“GaussianModel”（用于存储所有可优化的高斯的参数）。
    gaussians : GaussianModel 

    # 初始化场景对象的函数
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        - param path: Path to colmap scene main folder.
        - self：表示创建的Scene类实例本身
        - args : ModelParams：创建一个成员变量，类型为ModelParams（定义在“arguments/_init_.py”中，保存模型的参数和路径）
        - load_iteration：指定加载模型的迭代次数，默认为 None 表示新建训练。
        - shuffle：是否在训练前打乱相机列表，用于训练随机化
        - resolution_scales：控制渲染时的分辨率缩放比例，[1.0, 0.5] 表示可以以原分辨率和一半分辨率进行渲染。
        """
        self.model_path = args.model_path # 读取模型保存路径
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration: # 确定是否要加载一个训练好的模型的checkpoint
            if load_iteration == -1: # 特殊标记值
                # 自动寻找当前模型路径中保存的最大迭代号，即最新的 checkpoint。
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 初始化相机信息字典，后续用来存储相机视角数据。
        self.train_cameras = {}
        self.test_cameras = {}

        # 通过检查文件结构，判断输入数据集属于Colmap格式还是Blender格式，并调用对应的加载函数 
        # Colmap 在导出时，通常会生成一个 sparse/ 文件夹，来存储相机位姿和稀疏点云
        # Blender 渲染输出的数据集通常包含：transforms_train.json、transforms_val.json、transforms_test.json以及对应的渲染图像
        if os.path.exists(os.path.join(args.source_path, "sparse")): 
            # readColmapScene，解析并读取路径对应的图像、相机内参、外参等信息，最后返回一个 scene_info 对象
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            # readBlenderScene，读取 JSON 文件中的相机位姿、焦距、图像路径等信息，并打包成 scene_info 对象
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter: # 判断是否是首次训练，若是则需要新建一个训练场景
            # scene_info.ply_path ：数据集中提供的初始点云路径
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read()) # 将初始点云路径拷贝到当前模型文件夹下
            # 初始化相机列表，临时存储所有相机
            json_cams = []
            camlist = []
            # 让camlist包含列表中所有的相机视角。列表中每个元素都是一个Camera类实例字典，包含如id、旋转矩阵、平移量、焦距、图像路径等相机参数。
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                # 把相机对象（Camera 类实例）转换成可序列化的 JSON 格式字典，存入json_cams中
                json_cams.append(camera_to_JSON(id, cam)) 
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file) # 将 json_cams 列表序列化保存在cameras.json中

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # 多分辨率训练中保持一致的随机顺序
            random.shuffle(scene_info.test_cameras)  # 多分辨率训练中保持一致的随机顺序

        self.cameras_extent = scene_info.nerf_normalization["radius"] # 提取并保存场景归一化半径

        for resolution_scale in resolution_scales: # 根据不同分辨率构建相机列表
            print("Loading Training Cameras")
            # 根据当前分辨率比例，将 scene_info.train_cameras 转换成 CameraList 对象，并存入字典 self.train_cameras
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter: # 判断是否存在已训练的模型（loaded_iter 是否有值）
            # 当前路径中（iteration_xxx/point_cloud.ply）存在一个已训练的 3DGS 模型迭代结果，程序从文件中加载函数
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # 创建新高斯点云
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
