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

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
# 定义colmap支持的所有相机模型
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
# 基于CAMERA_MODEL创建一个字典,键为model_id，便于通过id查找相机模型
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
# 基于CAMERA_MODEL创建一个字典,键为model_name，便于通过name查找相机模型
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


# 实现四元数（w,x,y,z）到3×3旋转矩阵的转换
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# 对3×3旋转矩阵进行对称特征分解（必为实特征值），取最大特征值的特征向量，得到“最佳四元数”
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat # 平铺旋转阵，取出9个值
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    # np.linalg.eigh()遇到非对称的矩阵，只会读取下三角，并视为对称阵。
    eigvals, eigvecs = np.linalg.eigh(K) 
    # 取最大特征值对应的特征向量。其中，特征分解默认[x,y,z,w]，需要改变为四元数顺序[w,x,y,z]
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)] 
    # 利用q与-q表示相同旋转，保证四元数符号一致性。
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# 二进制解析原语。该函数读取文件中的二进制数据，根据提供的总字节数、格式字符串、字节序，来转换成对应数据类型。
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes) # 从当前文件指针位置开始，读取指定的字节数
    return struct.unpack(endian_character + format_char_sequence, data) # 用struct.unpack()解码字节流，返回一个元组

# 该函数是用来从“colmap导出的稀疏点云文件points3D_text（一个列表）”中，读取每个高斯点的参数（空间坐标xyz、颜色rgb、重投影误差error等）
def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid: # 先扫描一遍文件，统计点的数量num_points
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip() # 去除该行首尾空白符（包括空格、制表符、换行符等）
            if len(line) > 0 and line[0] != "#": # 过滤空行和注释行
                num_points += 1

    # 初始化存储数组：预先分配好三块连续内存。
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip() # 去除该行首尾空白符（包括空格、制表符、换行符等）
            if len(line) > 0 and line[0] != "#":
                elems = line.split() # 将字符串分割成字段列表
                # map()会对字符串数组中每一个元素调用函数，转变为浮点数。但map()返回的结果是个可迭代对象，不能直接np.array()变成numpy数组。
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                #依次填入数组
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

# 与read_points3D_text()类似，但是从二进制版点云文件points3D.bin中，读取所有高斯的参数。
def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid: # rb表示二进制读取模式
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            """
            Q：uint64，1×8字节————point3D_id
            ddd：double，3×8字节————xyz
            BBB：uint8，3×1字节————rgb
            d：double，1×8字节————error
            """
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            # 读取track_length参数，表示该点被多少个图像观测到。不保存，只移动指针。
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            # 读取接下来的track数组，长度为track_length，每个元素是一对整数 (image_id, point2D_idx)。不保存，只移动指针到下一个3d点的43字节。
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

# 从“colmap导出的相机参数文件cameras.txt”中，读取所有相机的内参，并构建一个Camera类型对象的字典
def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip() # 去除该行首尾空白符（包括空格、制表符、换行符等）
            if len(line) > 0 and line[0] != "#":
                elems = line.split() # 按照空格拆分line
                camera_id = int(elems[0])
                model = elems[1]
                # 训练代码目前只兼容PINHOLE相机，否则会抛出AssertionError
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                # PINHOLE相机的params包括fx,fy两个焦距（单位：像素）和cx,cy主点坐标（单位：焦距）
                # map()把每个字符串转换成浮点数迭代器，tuple()把迭代器变成元组，np.array()把元组变成numpy数组
                params = np.array(tuple(map(float, elems[4:])))
                # 创建相机对象，存储相机内参，camera_id是唯一键
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

# 从“colmap导出的二进制版文件cameras.bin”中，读取相机内参，返回一个camera类型对象的字典
def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0] # 先读取相机个数
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            # 从先前的相机列表字典中读取第二个：PINHOLE，取出model_name
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            # 从先前的相机列表字典中读取第二个：PINHOLE，取出num_params(如PINHOLE的num_params为4)
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params)) # 对params不需要map()和tuple()，是因为read_next_bytes中已经用struct.unpack()解码成元组了。
        assert len(cameras) == num_cameras # 循环结束后检查相机读取数量，确保文件没有截断或损坏
    return cameras

# 从“colmap导出的相机外参、图像观测数据文件image.txt”中读取图像的相机外参、其他图像数据，返回一个Image类型对象的字典
# image.txt中，每张图片对应两行数据。
def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()# 读取第一行：相机外参和相机信息
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5]))) # 读取四元数
                tvec = np.array(tuple(map(float, elems[5:8]))) # 读取平移向量
                camera_id = int(elems[8]) # 读取图片对应的相机编号
                image_name = elems[9]
                elems = fid.readline().split() # 读取紧接着的下一行：该图片的观测数据
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
