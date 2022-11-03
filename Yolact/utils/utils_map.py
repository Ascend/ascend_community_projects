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

import json
import os.path as osp
import os
import stat
import numpy as np
import pycocotools


class MakeJson:
    def __init__(self, path, coco_label_map):
        self.path = path
        self.data = []
        self.mask = []
        self.coco = {}

        for coco_id, real_id in coco_label_map.items():
            cl_id = real_id - 1
            self.coco[cl_id] = coco_id

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.data.append(
            {
                'image_id'      : int(image_id),
                'category_id'   : self.coco.get(int(category_id)),
                'bbox'          : bbox,
                'score'         : float(score)
            }
        )

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')

        self.mask.append(
            {
                'image_id'      : int(image_id),
                'category_id'   : self.coco.get(int(category_id)),
                'segmentation'  : rle,
                'score'         : float(score)
            }
        )

    def dump(self):
        dump_arguments = [
            (self.data, osp.join(self.path, "bbox_detections.json")),
            (self.mask, osp.join(self.path, "mask_detections.json"))
        ]

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL  # 注意根据具体业务的需要设置文件读写方式
        modes = stat.S_IWUSR | stat.S_IRUSR  # 注意根据具体业务的需要设置文件权限

        for data, path in dump_arguments:
            #with open(path, 'w') as f:
            with os.fdopen(os.open(path, flags, modes), 'w') as f:
                json.dump(data, f)
                


def prep_metrics(boxes, confs, classes, pred_masks, img_id, make_jsonn):
    classes    = list(np.array(classes, np.int32))
    confs      = list(np.array(confs, np.float32))
    for i in range(boxes.shape[0]):
        if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
            make_jsonn.add_bbox(img_id, classes[i], boxes[i, :], confs[i])
            make_jsonn.add_mask(img_id, classes[i], pred_masks[:, :, i], confs[i])
