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
import time
import stat
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, MxProtobufIn, InProtobufVector, StringVector

# The following belongs to the SDK Process
steam_manager_api = StreamManagerApi()
# init stream manager
ret = steam_manager_api.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()
# Mark start time
start = time.time()
# create streams by pipeline config file
# load  pipline
MODES = stat.S_IWUSR | stat.S_IRUSR
with os.fdopen(os.open("../pipeline/pre_post.pipeline", os.O_RDONLY, MODES), 'rb') as f:
    pipelineStr = f.read()
ret = steam_manager_api.CreateMultipleStreams(pipelineStr)
    # Print error message
if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()
        
print("load pipline done!")
TESTIMGS = 0
# Input object of streams -- detection target
PATH = "../test/data/val2017/"
TXT_PATH = "../test/" + "test_pre_post/"
if not os.path.exists(TXT_PATH):
    os.makedirs(TXT_PATH)
for item in os.listdir(PATH):
    img_path = os.path.join(PATH, item)
    print("file_path:", img_path)
    img_name = item.split(".")[0]
    img_txt = TXT_PATH + img_name + ".txt"
    if os.path.exists(img_txt):
        os.remove(img_txt)
    
    STREAMNAME = b'detection'
    INPLUGINID = 0
    dataInput = MxDataInput()
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(img_path, os.O_RDONLY, MODES), 'rb') as f:
        dataInput.data = f.read()
    uniqueId = steam_manager_api.SendData(STREAMNAME, INPLUGINID, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
    key_plugins = [b"mxpi_objectpostprocessor0"]
    vector = StringVector()
    for key in key_plugins:
        vector.push_back(key)
    result = steam_manager_api.GetProtobuf(STREAMNAME, 0, vector)
    if result.size() == 0:
        print("No object detected")
        continue
    if result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            result[0].errorCode, result[0].data.decode()))
        exit()
    # process data output from mxpi_objectpostprocessor plugin
    object_list = MxpiDataType.MxpiObjectList()
    object_list.ParseFromString(result[0].messageBuf)
    for obj in object_list.objectVec:
        with os.fdopen(os.open(img_txt, os.O_RDWR | os.O_CREAT, MODES), 'a+') as f:
            CONTENT = '{} {} {} {} {} {}'.format(obj.classVec[0].className,
                                                 round(obj.classVec[0].confidence, 4),
                                                 int(obj.x0),
                                                 int(obj.y0),
                                                 int(obj.x1),
                                                 int(obj.y1))
            f.write(CONTENT)
            f.write('\n')
    TESTIMGS += 1
end = time.time()
total_time = end - start
# Mark spend time
print("Total Images:%d" % TESTIMGS)
print("Spend time:%10.3f" % total_time)
print("FPS:%10.3f" % (TESTIMGS/total_time))
# Destroy All Streams
steam_manager_api.DestroyAllStreams()
