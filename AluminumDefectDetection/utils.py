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

import os
import numpy as np
import cv2

GRAY = 114


def preproc(img, img_size, swap=(2, 0, 1)):
    """Resize the input image."""
    if len(img.shape) == 3:
        padding_image = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * GRAY
    else:
        padding_image = np.ones(img_size, dtype=np.uint8) * GRAY

    ratio = min(img_size[0] / img.shape[0], img_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.uint8)
    top = int((int(img.shape[1] * ratio) - int(img.shape[0] * ratio)) / 2)
    padding_image[top: top + int(img.shape[0] * ratio), :int(img.shape[1] * ratio)] = resized_img

    return padding_image, ratio


def clip_coords(boxes, shape):
    boxes[0:4:2] = boxes[0:4:2].clip(0, shape[1])  # x1, x2
    boxes[1:4:2] = boxes[1:4:2].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[0] -= pad[0]  # x padding
    coords[2] -= pad[0]  # x padding
    coords[1] -= pad[1]  # y padding
    coords[3] -= pad[1]  # y padding
    coords[0] /= gain  # x padding
    coords[2] /= gain  # x padding
    coords[1] /= gain  # y padding
    coords[3] /= gain  # y padding
    clip_coords(coords, img0_shape)
    return coords


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def is_jpg(image_path):
    _, ending = os.path.splitext(image_path)
    if ending != ".jpg":
        return False
    return True


def is_legal(image_path):
    if not os.path.exists(image_path):
        print("The test image does not exist.")
        exit()
    if os.path.getsize(image_path) == 0:
        print("Error!The test image is empty.")
        exit()
    if not is_jpg(image_path):
        print("Please enter a JPG image")
        exit()
