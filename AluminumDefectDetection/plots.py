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

import platform
import os
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

colors = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10), (146, 204, 23),
          (61, 219, 134), (26, 147, 52), (0, 212, 187)]


def box_label(im, box, label='', color=(128, 128, 128), lw=3):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, (255, 255, 255),
                    thickness=tf, lineType=cv2.LINE_AA)
    return np.asarray(im)
