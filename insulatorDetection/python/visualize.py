#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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


import cv2
import webcolors

STANDARD_COLORS = ['red']

# All the class names of the detection target
OBJECT_LIST = ['insualtor']


def from_colorname_to_bgr(color):
    """
    convert color name to bgr value

    Args:
        color: color name

    Returns: bgr value

    """
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    """
    generate bgr list from color name list

    Args:
        list_color_name: color name list

    Returns: bgr list

    """
    standard = []
    standard.append(from_colorname_to_bgr(list_color_name[0]))
    return standard


def plot_one_box(origin_img, coord, cls_id, label=None, box_score=None, line_thickness=None):
    """
    plot one bounding box on image

    Args:
        origin_img: pending image
        coord: coordinate of bounding box
        label: class label name of the bounding box
        box_score: confidence score of the bounding box
        color: bgr color used to draw bounding box
        line_thickness: line thickness value when drawing the bounding box

    Returns: None

    """
    color_list = standard_to_bgr(STANDARD_COLORS)
    color = color_list[cls_id]
    tl = line_thickness or int(round(0.001 * max(origin_img.shape[0:2])))  # line thickness
    if tl < 1:
        tl = 1
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(origin_img, c1, c2, color=color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(box_score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(origin_img, c1, c2, color, -1)  # filled
        cv2.putText(origin_img, '{}: {:.0%}'.format(label, box_score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)