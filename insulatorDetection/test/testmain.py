# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()
# Mark start time
start = time.time()
# create streams by pipeline config file
# load  pipline
with open("../pipeline/detect.pipeline", 'rb') as f:
    pipelineStr = f.read()
ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
# Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()

COUNT = 0
# Input object of streams -- detection target
PATH = "./dataset/JPEGImages/"
RESULTPATH = "./test_result/"
if os.path.exists(RESULTPATH) != 1:
    os.makedirs("./test_result/")
for item in os.listdir(PATH):
    img_path = os.path.join(PATH, item)
    img_name = item.split(".")[0]
    img_txt = "./test_result/" + img_name + ".txt"
    if os.path.exists(img_txt):
        os.remove(img_txt)
    dataInput = MxDataInput()
    if os.path.exists(img_path) != 1:
        print("The test image does not exist.")

    with open(img_path, 'rb') as f:
        dataInput.data = f.read()

    STEAMNAME = b'detection'
    INPLUGINID = 0
    # Send data to streams by SendDataWithUniqueId()
    uniqueId = streamManagerApi.SendData(STEAMNAME, INPLUGINID, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Get results from streams by GetResultWithUniqueId()
    keys = [b"mxpi_objectpostprocessor0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)    
    inferResult = streamManagerApi.GetProtobuf(STEAMNAME, 0, keyVec)
    if inferResult[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            inferResult[0].errorCode, inferResult[0].data.decode()))
        exit()
    COUNT = COUNT + 1
    # get ObjectList
    if inferResult.size() == 0:
        continue
    img = cv2.imread(img_path)
    img_shape = img.shape
    bboxes = []
    object_list = MxpiDataType.MxpiObjectList()
    object_list.ParseFromString(inferResult[0].messageBuf)
    for objInfo in object_list.objectVec:
        bboxes = {'x0': int(objInfo.x0),
               'x1': int(objInfo.x1),
               'y0': int(objInfo.y0),
               'y1': int(objInfo.y1),
               'class': int(objInfo.classVec[0].classId),
               'class_name': objInfo.classVec[0].className,
               'confidence': round(objInfo.classVec[0].confidence, 4)}
        L1 = []
        L1.append(int(bboxes.get('x0')))
        L1.append(int(bboxes.get('x1')))
        L1.append(int(bboxes.get('y0')))
        L1.append(int(bboxes.get('y1')))
        L1.append(bboxes.get('confidence'))
        L1.append(bboxes.get('class_name'))        

        # save txt for results
        
        with os.fdopen(os.open(img_txt, os.O_RDWR|os.O_CREAT, MODES), "w")  as f:
            CONTENT = '{} {} {} {} {} {}'.format(L1[5], L1[4], L1[0], L1[2], L1[1], L1[3])
            f.write(CONTENT)
            f.write('\n')

end = time.time()
cost_time = end - start
# Mark spend time
print("Image COUNT:%d" % COUNT)
print("Spend time:%10.3f" % cost_time)
print("fps:%10.3f" % (COUNT/cost_time))
# Destroy All Streams
streamManagerApi.DestroyAllStreams()
