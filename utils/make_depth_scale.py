import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

# 该函数用于进行深度尺度对齐，计算单目深度图与colmap深度图之间的缩放尺度（scale）、偏移（offset）。
def get_scales(key, cameras, images, points3d_ordered, args):
    # images：colmap输出的图像字典（相机编号、外参、2d投影坐标xys、对应3d点索引）。
    image_meta = images[key]
    # cameras：colmap输出的相机字典（内参、height、width）。
    cam_intrinsic = cameras[image_meta.camera_id]

    # 根据图像索引key，在images_metas中提取出长度为特征点个数N的“3D点索引数组”。在全局点云中没有匹配到任何3D点的2D点，索引为-1
    pts_idx = images_metas[key].point3D_ids

    # 过滤掉无效的2D点，得到与索引数组等长度的“布尔数组”（False表示无效点）
    mask = pts_idx >= 0
    # pts_idx < len(points3d_ordered)为“是否在索引范围内”的布尔数组。
    # *=表示进行逻辑and操作，筛选出“既对应了有效3D点，也在索引范围内”的2D图像点。
    mask *= pts_idx < len(points3d_ordered)
    
    #只保留mask掩码筛选后的3D索引、2D像素坐标。
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        # 依靠合法索引找到场景中的3D点坐标。
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    # 把世界坐标系下的3D点转换到以相机为原点的坐标系下
    R = qvec2rotmat(image_meta.qvec)
    # 由于坐标转换公式为x_c = R * x_w， x_w为3×1列向量。而N×3的pts中单个3D点为3×1行向量，因此需要改变dot方向：dot(pts, R.T)。
    pts = np.dot(pts, R.T) + image_meta.tvec

    # pts[..., 2]为所有3D点的z深度，将其转为逆深度。
    invcolmapdepth = 1. / pts[..., 2]

    # 接下来读取“单目深度图”：
    # len(image_meta.name.split('.')[-1])为后缀长度（如"jpg"为3），n_remove为加上'.'的后缀长度（如".jpg"为4）。
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    # image_meta.name[:-n_remove]：去掉后缀，得到不带扩展名的“图像名”。
    # cv2.imread(..., cv2.IMREAD_UNCHANGED)：读取与该图像对应的深度图（保持原始位深）。
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None

    # ndim返回数组维度，一般深度图为灰度图像，形状为[H, W]。但如果invmonodepthmap存在多通道，则取单通道。
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    # 将深度图的16位深度转为[0, 1]浮点表示(8bit单精度浮点数)
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()


    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
