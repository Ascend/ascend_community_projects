#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':
    # Create a new StreamManager object and Initialize
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Building pipeline
    with open("./pipeline/crowdcount.pipeline", 'rb') as f:
        pipelineStr = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    dataInput = MxDataInput()
    if os.path.exists('./data/test.jpg') == 1:
        with open("./data/test.jpg", 'rb') as f:
            dataInput.data = f.read()
            testName = "'./data/test.jpg'"
    elif os.path.exists('./data/test.png') == 1:
        with open("./data/test.png", 'rb') as f:
            dataInput.data = f.read()
            testName = "'./data/test.png'"
    else:
        print("The test image does not exist.")

    STREAM_NAME = b'uav_crowdcounting'
    IN_PLUGIN_ID = 0
    uniqueId = streamManagerApi.SendData(STREAM_NAME, IN_PLUGIN_ID, dataInput)

    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_tensorinfer0"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)

    infer_raw = streamManagerApi.GetResult(STREAM_NAME, b'appsink0', key_vec)
    infer_result = infer_raw.metadataVec[0]
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)
    result_list = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr
                                , dtype=np.float32)
    print("Current Test: {}, Predicted Count: {}".format(testName, int(np.sum(result_list))))
    # reshape the result to a density map with a fixed size of 64*80
    vis_img = np.array(result_list).reshape(64, 80)
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    # expand the width and height of the predicted density map by 10 times
    vis_img = cv2.resize(vis_img, (int(vis_img.shape[1] * 10), int(vis_img.shape[0] * 10)), cv2.INTER_LINEAR)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    cv2.imwrite("vis_img.jpg", vis_img)
    print("Predicted Density Map Saved!")

    # destroy streams
    streamManagerApi.DestroyAllStreams()