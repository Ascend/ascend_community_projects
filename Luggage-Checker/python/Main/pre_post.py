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
import cv2

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

from visualize import plot_one_box

if __name__ == '__main__':
    steammanager_api = StreamManagerApi()
    # init stream manager
    ret = steammanager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("../pipeline/pre_post.pipeline", os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steammanager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    dataInput = MxDataInput()
    # It is best to use absolute path
    FILENAME = "../test_img/test.jpg"
    RESULTFILE = "../test_img/pre_post_bgr.jpg"
    if os.path.exists(FILENAME) != 1:
        print("The test image does not exist. Exit.")
        exit()
    with os.fdopen(os.open(FILENAME, os.O_RDONLY, MODES), 'rb') as f:
        dataInput.data = f.read()
    STEAMNAME = b'detection'
    INPLUGINID = 0
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
    cv2.imwrite(RESULTFILE, img)

    # destroy streams
    steammanager_api.DestroyAllStreams()

