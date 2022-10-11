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
import imghdr
import os
import time 
import cv2
import numpy as np
from PIL import Image
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

 

# 大型火灾.jpg                                      大型火灾场景
# CarInFlames-FireFighterHelmetCam4591.png          汽车失火场景
# CarInFlames-FireFighterHelmetCam6896.png          汽车失火烟雾场景
# HelmetCam2ndAlarmDwellingFire2361                 屋顶失火场景
# HelmetCam2ndAlarmDwellingFire179                  屋顶失火烟雾场景
# Ogdenhousefire7071                                房屋失火场景
# Ogdenhousefire459                                 房屋失火烟雾场景
# HelmetCam2ndAlarmDwellingFire1185.png             火灾场景拍摄不完整
# 夕阳.jpg                                          夕阳
# 山间道路.jpg                                      山间道路
# 城市夜灯.jpg                                      城市夜灯

TEST_PATH = '{图片所在文件夹路径}' # 若在fire_detection文件夹下可设置为./fire_detection/other
PICTURE = '{图片名称}' # 修改图片名称，如：大型火灾.jpg
TEST_PATH = TEST_PATH + PICTURE 
PIC_TYPE = imghdr.what(TEST_PATH)

min_image_size = 32
max_image_size = 8192

if os.path.exists(TEST_PATH) != 1:
    print("Failed to get the input picture. Please check it!")
    streamManagerApi.DestroyAllStreams()
    exit()
else:
    image = Image.open(TEST_PATH)
    if (image.format == 'JPEG' or image.format == 'PNG') != 1:
        print('input image only support jpg and png, curr format is {}.'.format(image.format))
        streamManagerApi.DestroyAllStreams()
        exit()
    elif image.width < min_image_size or image.width > max_image_size:
        print('input image width must in range [{}, {}], curr width is {}.'.format(
            min_image_size, max_image_size, image.width))
        streamManagerApi.DestroyAllStreams()
        exit()
    elif image.height < min_image_size or image.height > max_image_size:
        print('input image height must in range [{}, {}], curr height is {}.'.format(
            min_image_size, max_image_size, image.height))
        streamManagerApi.DestroyAllStreams()
        exit()


STREAM_NAME = b'classication'    # 流的名称
IN_PLUGIN_ID = 0   


# 输入为png图片则转换为jpg再进行读取，若为jpg直接读取
if PIC_TYPE == 'png':
    tmp_ = cv2.imread(TEST_PATH)
    TEST_PATH = TEST_PATH.replace('.png', '.jpg')
    cv2.imwrite(TEST_PATH, tmp_)
    dataInput = MxDataInput()

    with open(TEST_PATH, 'rb') as f:
        print("-----------------开始读取图片-----------------")
        dataInput.data = f.read()
        print("-----------------读取图片成功-----------------")

    os.remove(TEST_PATH)
else:
    dataInput = MxDataInput()

    with open(TEST_PATH, 'rb') as f:
        print("-----------------开始读取图片-----------------")
        dataInput.data = f.read()
        print("-----------------读取图片成功-----------------")

start_time = time.perf_counter()
# 目标输入插件Id，即appsrc元件的编号
uniqueId = streamManagerApi.SendData(STREAM_NAME, IN_PLUGIN_ID, dataInput) # SendData接口将图片数据发送给appsrc元件

start_time = time.perf_counter()   # 推理开始时间戳

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

end_time = time.perf_counter()

pre = prediction[0][0] * 100

if prediction[0][0] < 0.5:
    PREDICT = "Prediction: no fire"
else:
    PREDICT = "Prediction: fire"

        
# Destroy All Streams

print("-----------------Destroy All Streams-----------------")
streamManagerApi.DestroyAllStreams()
print(PREDICT)
print("耗时：", (end_time - start_time)*1000, "ms")
