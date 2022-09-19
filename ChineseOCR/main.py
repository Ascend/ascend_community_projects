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

import json
import os
import glob
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector

if __name__ == '__main__':
    # init stream manager
    pipeline_path = "chineseocr.pipeline"
    streamName = b'chineseocr'
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    inplugin_id = 0
    # Construct the input of the stream
    data_input = MxDataInput()

    output_path = os.path.dirname(os.path.realpath(__file__))
    output_exer = os.path.join(output_path, 'output')
    if not os.path.exists(output_exer):
        os.makedirs(output_exer)
    for index, img_path in enumerate(glob.glob(os.path.join(output_path, '6/*.jpg'))):
        text_label = img_path.replace('jpg', 'txt')
        with open(img_path, 'rb') as fp:
            data_input.data = fp.read()
        unique_id = stream_manager_api.SendData(streamName, b'appsrc0', data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_textgenerationpostprocessor0')
        infer_result = stream_manager_api.GetProtobuf(streamName, inplugin_id, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()
        result = MxpiDataType.MxpiTextsInfoList()
        result.ParseFromString(infer_result[0].messageBuf)
        content_pic = str(result.textsInfoVec[0].text)
        print(content_pic[2:-2])
        with open(text_label, 'r', encoding='utf-8') as f:
            content = ""
            for i in f.readlines():
                content += i.strip()

            with open(os.path.join(output_exer, f'{index}.txt'), 'w', encoding='utf-8') as wf:
                wf.write(content)
            with open(os.path.join(output_exer, f'{index}_pic.txt'), 'w', encoding='utf-8') as wf:
                wf.write(content_pic[2:-2])

    # destroy streams
    stream_manager_api.DestroyAllStreams()
