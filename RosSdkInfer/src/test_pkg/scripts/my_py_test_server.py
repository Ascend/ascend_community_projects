#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
"""

import sys
import re
import json
import os
import signal
import datetime
import threading
import random
import base64
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import rospy
from test_pkg.srv import MyResult, MyResultResponse

STREAM_MANAGER_API = {}


def pic_infer(pic_data):
    global STREAM_MANAGER_API
    # Construct the input of the stream
    data_input = MxDataInput()
    data_input.data = pic_data

    # Inputs data to a specified stream based on stream_name.
    stream_name = b'classification+detection'
    in_plugin_id = 0
    unique_id = STREAM_MANAGER_API.SendDataWithUniqueId(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying stream_name and unique_id.
    infer_result = STREAM_MANAGER_API.GetResultWithUniqueId(stream_name, unique_id, 3000)
    if infer_result.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            infer_result.errorCode, infer_result.data.decode()))
        exit()

    # print the infer result
    ret_str = infer_result.data.decode()
    print("server infer the result:\n")
    print(ret_str)
    return ret_str


def sdk_call_back(req):
	# 显示请求数据
    pic_data = base64.b64decode(req.imageStr)
    ret_str = pic_infer(pic_data)

	# 反馈数据
    return MyResultResponse(ret_str)


def sdk_server():
	# ROS节点初始化
    rospy.init_node('sdk_server')

	# 创建一个名为/show_person的server，注册回调函数sdk_call_back
    s = rospy.Service('/sdk_infer', MyResult, sdk_call_back)

	# 循环等待回调函数
    print("Ready to get image string.")
    rospy.spin()


def sdk_init():
    # init stream manager
    global STREAM_MANAGER_API
    STREAM_MANAGER_API = StreamManagerApi()
    ret = STREAM_MANAGER_API.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        STREAM_MANAGER_API.DestroyAllStreams()
        exit()

    # create streams by pipeline config file
    pipeline_path = b"/home/HwHiAiUser/catkin_ws/src/test_pkg/scripts/pipeline/Sample.pipeline"
    with open(pipeline_path, 'rb') as f:
        pipeline_str = f.read()

    ret = STREAM_MANAGER_API.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        STREAM_MANAGER_API.DestroyAllStreams()
        exit()

if __name__ == "__main__":
    sdk_init()
    sdk_server()



