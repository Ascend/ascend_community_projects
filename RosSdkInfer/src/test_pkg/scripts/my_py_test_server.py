#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    dataInput = MxDataInput()
    dataInput.data = pic_data

    # Inputs data to a specified stream based on streamName.
    streamName = b'classification+detection'
    inPluginId = 0
    uniqueId = STREAM_MANAGER_API.SendDataWithUniqueId(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = STREAM_MANAGER_API.GetResultWithUniqueId(streamName, uniqueId, 3000)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    # print the infer result
    retStr = inferResult.data.decode()
    print("server infer the result:\n")
    print(retStr)
    return retStr


def sdkCallback(req):
	# 显示请求数据
    # rospy.loginfo("resutl:%s", req.inferResult)
    pic_data = base64.b64decode(req.imageStr)
    retStr = pic_infer(pic_data)

	# 反馈数据
    return MyResultResponse(retStr)

def sdk_server():
	# ROS节点初始化
    rospy.init_node('sdk_server')

	# 创建一个名为/show_person的server，注册回调函数sdkCallback
    s = rospy.Service('/sdk_infer', MyResult, sdkCallback)

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
    pipelinePath = b"/home/HwHiAiUser/hhy/catkin_ws/src/test_pkg/scripts/pipeline/Sample.pipeline"
    with open(pipelinePath, 'rb') as f:
        pipelineStr = f.read()

    ret = STREAM_MANAGER_API.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        STREAM_MANAGER_API.DestroyAllStreams()
        exit()

if __name__ == "__main__":
    sdk_init()
    sdk_server()



