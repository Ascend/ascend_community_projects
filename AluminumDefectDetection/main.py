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
import json
import cv2
from StreamManagerApi import StreamManagerApi, MxDataInput
import numpy as np
from plots import box_label, colors
from utils import preprocess, scale_coords, xyxy2xywh

names = ['non_conduct', 'abrasion_mark', 'corner_leak', 'orange_peel', 'leak', 'jet_flow', 'paint_bubble', 'pit',
         'motley', 'dirty_spot']
if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/AlDefectDetection.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    ORI_IMG_PATH = "test.jpg"
    if not os.path.exists(ORI_IMG_PATH):
        print("The test image does not exist.")
        exit()
    if os.path.getsize(ORI_IMG_PATH) == 0:
        print("Error!The test image is empty.")
        exit()

    # read image
    ori_img = cv2.imread(ORI_IMG_PATH)
    h0, w0 = ori_img.shape[:2]
    r = 640 / max(h0, w0)  # ratio

    input_shape = (640, 640)
    pre_img = preprocess(ori_img, input_shape)[0]
    pre_img = np.ascontiguousarray(pre_img)
    PRE_IMG_PATH = "pre_" + ORI_IMG_PATH
    cv2.imwrite(PRE_IMG_PATH, pre_img)

    with open(PRE_IMG_PATH, 'rb') as f:
        dataInput.data = f.read()

    # Inputs data to a specified stream based on streamName.
    STREAMNAME = b'classification+detection'
    INPLUGINID = 0
    uniqueId = streamManagerApi.SendDataWithUniqueId(STREAMNAME, INPLUGINID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResultWithUniqueId(STREAMNAME, uniqueId, 10000)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    results = json.loads(inferResult.data.decode())
    gn = np.array(ori_img.shape)[[1, 0, 1, 0]]

    bboxes = []
    classVecs = []
    # draw the result and save image
    for info in results['MxpiObject']:
        bboxes.append([int(info['x0']), int(info['y0']), int(info['x1']), int(info['y1'])])
        classVecs.append(info["classVec"])
    for (xyxy, classVec) in zip(bboxes, classVecs):
        xyxy = scale_coords(pre_img.shape[:2], np.array(xyxy), ori_img.shape[:2])
        xywh = (xyxy2xywh(xyxy.reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
        print(classVec)
        label = f'{classVec[0]["className"]} {classVec[0]["confidence"]:.4f}'
        save_img = box_label(ori_img, xyxy, label, color=colors[names.index(classVec[0]["className"])])

    cv2.imwrite('./result_' + ORI_IMG_PATH, save_img)

    ######################################################################################

    # print the infer result
    print(inferResult.data.decode())

    # destroy streams
    streamManagerApi.DestroyAllStreams()
