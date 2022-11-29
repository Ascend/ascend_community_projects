# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import cv2


def get_color_table(cla_um, seetmp=2):
    random.seed(seetmp)
    color_tmp_table = {}
    for i in range(cla_um):
        color_tmp_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_tmp_table


def plot_one_box(
        img_tmp,
        coor_tmp,
        label_tmp=None,
        color_tmp=None,
        line_tmp_thick=None):

    tl = line_tmp_thick or int(
        round(0.002 * max(img_tmp.shape[0:2])))  # line thickness
    color_tmp = color_tmp or [random.randint(0, 255) for _ in range(3)]
    c1_tmp, c2_tmp = (int(coor_tmp[0]), int(
        coor_tmp[1])), (int(coor_tmp[2]), int(coor_tmp[3]))
    cv2.rectangle(img_tmp, c1_tmp, c2_tmp, color_tmp, thickness=tl)
    if label_tmp:
        tf = max(tl - 1, 1)  # font thickness
        t_tmp_size = cv2.getTextSize(
            label_tmp, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2_tmp = c1_tmp[0] + t_tmp_size[0], c1_tmp[1] - t_tmp_size[1] - 3
        cv2.rectangle(img_tmp, c1_tmp, c2_tmp, color_tmp, -1)  # filled
        cv2.putText(img_tmp,
                    label_tmp,
                    (c1_tmp[0],
                     c1_tmp[1] - 2),
                    0,
                    float(tl) / 3,
                    [0,
                        0,
                        0],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
