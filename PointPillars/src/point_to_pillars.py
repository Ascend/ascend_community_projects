"""
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import os
import struct
import numpy as np
import torch
import fire


def read_lidar_info(file_path):
    size = os.path.getsize(file_path)
    point_num = int(size / 16)
    assert point_num * 16 == size

    lidar_pt_list = np.zeros((point_num, 4))
    with open(file_path, 'rb') as f:
        for i in range(point_num * 4):
            data = f.read(4)
            val = struct.unpack('f', data)
            row = int(i / 4)
            col = i % 4
            lidar_pt_list[row][col] = val[0]
    return lidar_pt_list


def points_to_voxel_kernel(points,
                           voxel_size,
                           coors_range,
                           num_points_per_voxel,
                           coor_to_voxelidx,
                           voxels,
                           coors,
                           max_points=100,
                           max_voxels=12000):
    point_cnt = points.shape[0]
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(point_cnt):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(points,
                    voxel_size,   # (0.16, 0.16, 4.0)
                    coors_range,  # (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
                    max_points=100,
                    max_voxels=12000):
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = points_to_voxel_kernel(
        points, voxel_size, coors_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


def get_sub_shaped(coors):
    x_sub = coors[:, 0] * 0.16 + 0.08
    y_sub = coors[:, 1] * 0.16 - 39.6
    x_sub_shaped = np.zeros((12000, 100))
    y_sub_shaped = np.zeros((12000, 100))
    for i in range(0, 100):
        x_sub_shaped[:12000, i] = x_sub
        y_sub_shaped[:12000, i] = y_sub
    x_sub_shaped = torch.as_tensor(x_sub_shaped).unsqueeze(0).unsqueeze(0).numpy()
    y_sub_shaped = torch.as_tensor(y_sub_shaped).unsqueeze(0).unsqueeze(0).numpy()
    return x_sub_shaped, y_sub_shaped


def pillar_expand(voxel):
    pillar = np.zeros((12000, 100))
    pillar_len = voxel.shape[0]
    for i in range(0, 100):
        pillar[:pillar_len, i] = voxel[:, i]
    return pillar


def cnt_expand(num_points_per_vexols):
    cnt = np.zeros((12000))
    cnt_len = num_points_per_vexols.shape[0]
    cnt[:cnt_len] = num_points_per_vexols
    return cnt


def coors_expand(coor):
    coors = np.zeros((12000, 3))
    coors_len = coor.shape[0]
    coors[:coors_len, :] = coor[:, :]
    return coors


def get_mask(actual_num_numpy, max_num, axis=0):
    actual_num = torch.as_tensor(actual_num_numpy)
    actual_num = torch.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    paddings_indicator = paddings_indicator.permute(0, 2, 1)
    paddings_indicator = paddings_indicator.unsqueeze(1)
    return paddings_indicator


def generate(file_dir="../data/test/"):
    point = read_lidar_info(f"{file_dir}/point.bin")
    voxel_size = [0.16, 0.16, 4.0]
    coors_range = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    voxels, coor, num_points_per_vexols = points_to_voxel(point, voxel_size, coors_range)
    coors = coors_expand(coor)
    print(point.shape)
    print(voxels.shape)
    print(coors.shape)
    print(num_points_per_vexols.shape)
    print(voxels)
    pillar_x = torch.as_tensor(pillar_expand(voxels[:, :, 0])).unsqueeze(0).unsqueeze(0).numpy().astype(np.float16)
    pillar_y = torch.as_tensor(pillar_expand(voxels[:, :, 1])).unsqueeze(0).unsqueeze(0).numpy().astype(np.float16)
    pillar_z = torch.as_tensor(pillar_expand(voxels[:, :, 2])).unsqueeze(0).unsqueeze(0).numpy().astype(np.float16)
    pillar_i = torch.as_tensor(pillar_expand(voxels[:, :, 3])).unsqueeze(0).unsqueeze(0).numpy().astype(np.float16)
    x_sub_shaped, y_sub_shaped = get_sub_shaped(coors)
    x_sub_shaped = x_sub_shaped.astype(np.float16)
    y_sub_shaped = y_sub_shaped.astype(np.float16)
    num_points_per_pillar = torch.as_tensor(cnt_expand(num_points_per_vexols)).unsqueeze(0).numpy().astype(np.float16)
    num_points_for_pillar = torch.as_tensor(pillar_x).size()[3]
    mask = get_mask(num_points_per_pillar, num_points_for_pillar, axis=0).numpy().astype(np.float16)
    print(pillar_x.shape)
    print(pillar_y.shape)
    print(pillar_z.shape)
    print(pillar_i.shape)
    print(x_sub_shaped.shape)
    print(y_sub_shaped.shape)
    print(num_points_per_pillar.shape)
    print(mask.shape)

    pillar_x.tofile(f"{file_dir}/pillar_x.bin")
    pillar_y.tofile(f"{file_dir}/pillar_y.bin")
    pillar_z.tofile(f"{file_dir}/pillar_z.bin")
    pillar_i.tofile(f"{file_dir}/pillar_i.bin")
    x_sub_shaped.tofile(f"{file_dir}/x_sub_shaped.bin")
    y_sub_shaped.tofile(f"{file_dir}/y_sub_shaped.bin")
    num_points_per_pillar.tofile(f"{file_dir}/num_points_per_pillar.bin")
    mask.tofile(f"{file_dir}/mask.bin")

    np.save(f"{file_dir}/coor.npy", coors)


if __name__ == '__main__':
    fire.Fire()
