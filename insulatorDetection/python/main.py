#!/usr/bin/env python
# coding=utf-8

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
import stat
import time
import webcolors
import cv2
from cv2 import getTickCount, getTickFrequency
from PIL import Image
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


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



if __name__ == '__main__':
    steammanager_api = StreamManagerApi()
    # init stream manager
    ret = steammanager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("../pipeline/detect.pipeline", os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steammanager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    # It is best to use absolute path
    FILENAME = "../data/22.jpg"
    RESULTFILE = "../data/output.jpg"
    
    if os.path.exists(FILENAME) != 1:
        print("The test image does not exist. Exit.")
        exit()
    image = Image.open(FILENAME)
    if image.format != "JPEG" and image.format != "JPG":
        print("the image is not JPG format")
        exit()   
                
    with os.fdopen(os.open(FILENAME, os.O_RDONLY, MODES), 'rb') as f:
        dataInput.data = f.read()
    STEAMNAME = b'detection'
    INPLUGINID = 0
    t1 = time.time()
    uniqueId = steammanager_api.SendData(STEAMNAME, INPLUGINID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    result = steammanager_api.GetProtobuf(STEAMNAME, 0, keyVec)
    if result.size() == 0:
        print("No object detected")
        img = cv2.imread(FILENAME)
        cv2.imwrite(RESULTFILE, img)
        exit()
    if result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            result[0].errorCode, result[0].data.decode()))
        exit()
    # process data output from mxpi_objectpostprocessor plugin
    object_list = MxpiDataType.MxpiObjectList()
    object_list.ParseFromString(result[0].messageBuf)
    bounding_boxes = []
    for obj in object_list.objectVec:
        print(obj)
        box = {'x0': int(obj.x0),
               'x1': int(obj.x1),
               'y0': int(obj.y0),
               'y1': int(obj.y1),
               'class': int(obj.classVec[0].classId),
               'class_name': obj.classVec[0].className,
               'confidence': round(obj.classVec[0].confidence, 4)}
        bounding_boxes.append(box)  
    img = cv2.imread(FILENAME)  
    # draw each bounding box on the original image
    for box in bounding_boxes:
        class_id = box.get('class')
        class_name = box.get('class_name')
        score = box.get('confidence')
        plot_one_box(img,
                     [box.get('x0'),
                      box.get('y0'),
                      box.get('x1'),
                      box.get('y1')],
                     cls_id=class_id,
                     label=class_name,
                     box_score=score)
    print("fps:", 1/(time.time()-t1))
    cv2.imwrite(RESULTFILE, img)
    # destroy streams
    steammanager_api.DestroyAllStreams()

