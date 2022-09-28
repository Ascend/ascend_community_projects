"""
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
"""

import json
import os
import time 
import cv2
import numpy as np
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import MxpiDataType_pb2 as MxpiDataType

# The following belongs to the SDK Process
streamManagerApi = StreamManagerApi()
# init stream manager
ret = streamManagerApi.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()
else:
    print("-----------------创建流管理StreamManager并初始化-----------------")

# create streams by pipeline config file
# load pipline
with open("./pipeline/fire.pipeline", 'rb') as f:
    print("-----------------正在读取读取pipeline-----------------")
    pipelineStr = f.read()
    print("-----------------成功读取pipeline-----------------")

ret = streamManagerApi.CreateMultipleStreams(pipelineStr)

# Print error message
if ret != 0:
    print("-----------------未能成功创建流-----------------")
    print("-----------------Failed to create Stream, ret=%s-----------------" % str(ret) )
else:
    print("-----------------成功创建流-----------------")
    print("-----------------Create Stream Successfully, ret=%s-----------------" % str(ret) )
# Stream name

# 读取图片

if os.path.exists('./data/test.jpg') != 1:
    print("-----------------未能读取图片-----------------")
    print("-----------------The test image does not exist.-----------------")

STREAM_NAME = b'classication'    # 流的名称
IN_PLUGIN_ID = 0    
TEST_PATH = "./data/test"

# 统计时间、图片张数
TOTAL_TIME = 0
PIC_NUM = 0
RIGHT_NUM = 0

for path_1 in os.listdir(TEST_PATH):
    dataInput = MxDataInput()
    path_2 = os.path.join(TEST_PATH, path_1)
    for file_ in os.listdir(path_2):

        file__ = os.path.join(path_2, file_)
        tmp_ = cv2.imread(file__)
        print(file__)
        file__ = file__.replace('.png', '.jpg')
        cv2.imwrite(file__, tmp_)
        with open(file__, 'rb') as f:
            print("-----------------开始读取图片-----------------")
            dataInput.data = f.read()
            print("-----------------读取图片成功-----------------")
        os.remove(file__)
        # 发送数据
        start_time = time.perf_counter()   # 推理开始时间戳
        # 目标输入插件Id，即appsrc元件的编号
        uniqueId = streamManagerApi.SendData(STREAM_NAME, IN_PLUGIN_ID, dataInput) # SendData接口将图片数据发送给appsrc元件

        if uniqueId < 0:
            print("-----------------数据未能发送至流-----------------")
            print("-----------------Failed to send data to stream.-----------------")
            exit()
        else:
            print("-----------------数据成功发送至流-----------------")

        # 获取数据
        keys = [b"mxpi_tensorinfer0"] # 设置GetProtobuf的MxProtobufIn列表
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)

        print("-----------------从流获取推理结果-----------------")

        infer_result = streamManagerApi.GetProtobuf(STREAM_NAME, 0, keyVec)  # 从流中取出对应插件的输出数据

        if infer_result.size() == 0:
            print("-----------------推理结果null-----------------")
            print("-----------------infer_result is null-----------------")
            exit()

        if infer_result[0].errorCode != 0:
            print("-----------------推理结果error-----------------")
            print("-----------------GetProtobuf error. errorCode=%d-----------------" % (
                infer_result[0].errorCode))
            exit()

        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[0].messageBuf)
        prediction = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float32)
        prediction_shape = tensorList.tensorPackageVec[0].tensorVec[0].tensorShape
        prediction = np.reshape(prediction, prediction_shape)
        
        if prediction[0][0] < 0.5:
            print("predict：no fire")
        else:
            print("predict：fire")
        
        end_time = time.perf_counter()                  # 推理结束时间戳
        sigTime = (end_time - start_time) * 1000    # 单张图片好费时间
        if(sigTime >= 40):  
            print("singal pic time out")
        TOTAL_TIME = TOTAL_TIME + sigTime           # 总耗费时间
        PIC_NUM = PIC_NUM + 1                           # 图片数量计数

        print("耗时时间：", str(sigTime), "ms")

        if '0_nofire' in file__ and prediction[0][0] < 0.5:
            RIGHT_NUM = RIGHT_NUM + 1
        if '1_fire' in file__ and prediction[0][0] >= 0.5:
            RIGHT_NUM = RIGHT_NUM + 1
        
# Destroy All Streams

print("-----------------Destroy All Streams-----------------")
streamManagerApi.DestroyAllStreams()
print("精度：", RIGHT_NUM/PIC_NUM*100, "%")
print("总耗时：", TOTAL_TIME, "ms   总图片数：", PIC_NUM)
print("平均单张耗时：", TOTAL_TIME / PIC_NUM, "ms")

# *******************************************************************

