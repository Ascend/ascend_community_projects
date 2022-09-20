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
from StreamManagerApi import *

from plots import Annotator, colors
from utils import *

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
    ori_img_path = "test.jpg"
    if not os.path.exists(ori_img_path):
        print("The test image does not exist.")
        exit()
    if os.path.getsize(ori_img_path) == 0:
        print("Error!The test image is empty.")
        exit()

    # read image
    ori_img = cv2.imread(ori_img_path)
    h0, w0 = ori_img.shape[:2]
    img_size = 640
    r = img_size / max(h0, w0)  # ratio

    input_shape = (640, 640)
    pre_img = letterbox(ori_img, input_shape)[0]
    pre_img = np.ascontiguousarray(pre_img)
    pre_img_path = "pre_" + ori_img_path
    cv2.imwrite(pre_img_path, pre_img)

    with open(pre_img_path, 'rb') as f:
        dataInput.data = f.read()

    # Inputs data to a specified stream based on streamName.
    streamName = b'classification+detection'
    inPluginId = 0
    uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 10000)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    results = json.loads(inferResult.data.decode())
    gn = np.array(ori_img.shape)[[1, 0, 1, 0]]

    annotator = Annotator(ori_img, line_width=3, example=str(names))
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
        annotator.box_label(xyxy, label, color=colors(names.index(classVec[0]["className"]), False))

    save_img = annotator.result()
    cv2.imwrite('./result_' + ori_img_path, save_img)

    ######################################################################################

    # print the infer result
    print(inferResult.data.decode())

    # destroy streams
    streamManagerApi.DestroyAllStreams()
