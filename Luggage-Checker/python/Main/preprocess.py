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
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padding_image[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img

    padding_image = padding_image.transpose(swap)
    padding_image = np.ascontiguousarray(padding_image, dtype=np.float32)
    return padding_image, ratio