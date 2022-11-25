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
from math import cos, sin
from mayavi import mlab
import numpy as np
import torch
import fire


def box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    # need to convert box_encodings to z-bottom format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = np.split(box_encodings, 8, axis=-1)
    else:
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, 7, axis=-1)
    za = za + ha / 2
    diagonal = np.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya

    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.Tensor(np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1))


def nms_op_kernel(dets, thresh=0.01, eps=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4].numpy()
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    nms_order = scores.argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    index_to_keep = []
    for _i in range(ndets):
        i = nms_order[_i] 
        if suppressed[
                i] == 1: 
            continue
        index_to_keep.append(i)
        for _j in range(_i + 1, ndets):
            j = nms_order[_j]
            if suppressed[j] == 1:
                continue
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return index_to_keep


def nms_op(boxes, scores, pre_maxsize=None):
    nms_order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        nms_order = nms_order[:pre_maxsize]
    boxes = boxes[nms_order].contiguous()
    index_to_keep = nms_op_kernel(boxes)
    return index_to_keep


def limit_period(val, offset=0.5, period=np.pi):
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def generate_anchors(feature_size,
                     anchor_range,
                     sizes,
                     rotations,
                     dtype):
    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        (anchor_range[2] + anchor_range[5]) / 2, anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    length = len(rets)
    for i in range(length):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def get_predict_result(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
    bbox_cls_pred = bbox_cls_pred.reshape(-1, 1)
    bbox_pred = bbox_pred.reshape(-1, 7)
    bbox_dir_cls_pred = bbox_dir_cls_pred.reshape(-1, 2)
    anchors = anchors.reshape(-1, 7)
    bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
    bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

    inds = bbox_cls_pred.max(1)[0].topk(100)[1]
    bbox_cls_pred = bbox_cls_pred[inds]
    bbox_pred = bbox_pred[inds]
    bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
    anchors = anchors[inds]

    bbox_pred = box_decode(bbox_pred, anchors)

    bbox_2d_xy = bbox_pred[:, [0, 1]]
    bbox_2d_wl = bbox_pred[:, [3, 4]]
    bbox_pred2d = torch.cat([bbox_2d_xy - bbox_2d_wl / 2,
                             bbox_2d_xy + bbox_2d_wl / 2,
                             bbox_cls_pred], dim=-1)
    ret_bboxes, ret_labels, ret_scores = [], [], []
    for i in range(1):
        cur_bbox_cls_pred = bbox_cls_pred[:, i]
        score_inds = cur_bbox_cls_pred > 0.1
        if score_inds.sum() == 0:
            continue

        cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
        cur_bbox_pred2d = bbox_pred2d[score_inds]
        cur_bbox_pred = bbox_pred[score_inds]
        cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
        
        keep_inds = nms_op(boxes=cur_bbox_pred2d, 
                                scores=cur_bbox_cls_pred, 
                                pre_maxsize=None)

        cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
        cur_bbox_pred = cur_bbox_pred[keep_inds]
        cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
        cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred)
        cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

        ret_bboxes.append(cur_bbox_pred)
        ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
        ret_scores.append(cur_bbox_cls_pred)

    if len(ret_bboxes) == 0:
        return [], [], []
    ret_bboxes = torch.cat(ret_bboxes, 0)
    ret_labels = torch.cat(ret_labels, 0)
    ret_scores = torch.cat(ret_scores, 0)
    cnt = 0
    for i in range(50):
        cnt += 1
        if ret_scores[i] < 0.6:
            cnt -= 1
            break
    final_inds = ret_scores.topk(cnt)[1]
    ret_bboxes = ret_bboxes[final_inds]
    ret_labels = ret_labels[final_inds]
    ret_scores = ret_scores[final_inds]
    result = {
        'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
        'labels': ret_labels.detach().cpu().numpy(),
        'scores': ret_scores.detach().cpu().numpy()
    }
    return result


def get_box_points(box_list):
    box_points_list = []
    length = len(box_list)
    for cnt in range(length):
        box = box_list[cnt]
        box_points = np.zeros((8, 3))
        x = float(box[0])
        y = float(box[1])
        z = float(box[2])
        width = float(box[3])
        long = float(box[4])
        depth = float(box[5])
        theta = np.pi / 2 - box[6]
        box_points[0] = [x + long / 2 * cos(theta) + width / 2 * sin(theta),
                         y + long / 2 * sin(theta) - width / 2 * cos(theta),
                         z + depth / 2]
        box_points[3] = [x + long / 2 * cos(theta) + width / 2 * sin(theta),
                         y + long / 2 * sin(theta) - width / 2 * cos(theta), 
                         z - depth / 2]

        box_points[1] = [x + long / 2 * cos(theta) - width / 2 * sin(theta), 
                         y + width / 2 * cos(theta) + long / 2 * sin(theta),
                         z + depth / 2]
        box_points[2] = [x + long / 2 * cos(theta) - width / 2 * sin(theta),
                         y + width / 2 * cos(theta) + long / 2 * sin(theta),
                         z - depth / 2]

        box_points[5] = [2 * x - (x + long / 2 * cos(theta) + width / 2 * sin(theta)),
                         2 * y - (y + long / 2 * sin(theta) - width / 2 * cos(theta)), 
                         z + depth / 2]
        box_points[6] = [2 * x - (x + long / 2 * cos(theta) + width / 2 * sin(theta)),
                         2 * y - (y + long / 2 * sin(theta) - width / 2 * cos(theta)), 
                         z - depth / 2]

        box_points[4] = [2 * x - (x + long / 2 * cos(theta) - width / 2 * sin(theta)),
                         2 * y - (y + width / 2 * cos(theta) + long / 2 * sin(theta)),
                         z + depth / 2]
        box_points[7] = [2 * x - (x + long / 2 * cos(theta) - width / 2 * sin(theta)),
                         2 * y - (y + width / 2 * cos(theta) + long / 2 * sin(theta)),
                         z - depth / 2]

        box_points_list.append(box_points)

    return np.array(box_points_list)


def draw_box(box_point_list):
    length = len(box_point_list)
    for cnt in range(length):
        for k in range(0, 4):
            box_point = box_point_list[cnt]
            i, j = k, (k + 1) % 4
            mlab.plot3d([box_point[i, 0], box_point[j, 0]],
                        [box_point[i, 1], box_point[j, 1]],
                        [box_point[i, 2], box_point[j, 2]])
            i, j = k + 4, (k + 3) % 4 + 4
            mlab.plot3d([box_point[i, 0], box_point[j, 0]],
                        [box_point[i, 1], box_point[j, 1]],
                        [box_point[i, 2], box_point[j, 2]])
            i , j = k, k + 4
            mlab.plot3d([box_point[i, 0], box_point[j, 0]],
                        [box_point[i, 1], box_point[j, 1]],
                        [box_point[i, 2], box_point[j, 2]])


def point_show(points, box):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z,
                        z,  # Values used for Color
                        mode="point",
                        colormap='spectral',
                        figure=fig,
                        )
    box_point = get_box_points(box)
    draw_box(box_point)
    
    mlab.show()


def get_result(file_dir='../result/test/'):
    anchors = torch.as_tensor(generate_anchors(feature_size=[1, 248, 216],
                                               anchor_range = [0, -39.68, -3, 69.12, 39.68, 1],
                                               sizes=[1.6, 3.9, 1.56],
                                               rotations=[0, np.pi / 2],
                                               dtype=np.float32).reshape((248, 216, 1, 2, 7)))
    bbox_cls_pred = torch.as_tensor(np.fromfile(f"{file_dir}/cls.bin", dtype=np.float32)
                         .reshape((1, 248, 216, 2)))
    bbox_pred = torch.as_tensor(np.fromfile(f"{file_dir}/box.bin", dtype=np.float32)
                     .reshape((1, 248, 216, 14)))
    bbox_dir_cls_pred = torch.as_tensor(np.fromfile(f"{file_dir}/dir.bin", dtype=np.float32)
                             .reshape((1, 248, 216, 4)))
    result = get_predict_result(bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors)
    return result


def viewer(file_dir='../result/test/'):
    if os.path.exists(file_dir):
        file = file_dir + "point.bin"
        if os.path.exists(file):
            points = np.fromfile(file, dtype=np.float32).reshape([-1, 4])
            result = get_result(file_dir)
            print(result)
            box = result['lidar_bboxes']
            boxes = np.array(box)
            boxes.tofile(f"{file_dir}/result.bin")
            point_show(points, box)
        else:
            print(f"file : {file} does not exist")
    else:
        print(f"path : {file_dir} does not exist")


if __name__ == '__main__':
    fire.Fire()
