#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from test_pkg.srv import MyResult, MyResultResponse
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

streamManagerApi = {}

def picInfer(picData):
    global streamManagerApi
    # Construct the input of the stream
    dataInput = MxDataInput()
    dataInput.data = picData

    # Inputs data to a specified stream based on streamName.
    streamName = b'classification+detection'
    inPluginId = 0
    uniqueId = streamManagerApi.SendDataWithUniqueId(streamName, inPluginId, dataInput)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResultWithUniqueId(streamName, uniqueId, 3000)
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
    picData = base64.b64decode(req.imageStr)
    retStr = picInfer(picData)

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
    global streamManagerApi
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        streamManagerApi.DestroyAllStreams()
        exit()

    # create streams by pipeline config file
    pipelinePath = b"/home/HwHiAiUser/hhy/catkin_ws/src/test_pkg/scripts/pipeline/Sample.pipeline"
    with open(pipelinePath, 'rb') as f:
        pipelineStr = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        streamManagerApi.DestroyAllStreams()
        exit()

if __name__ == "__main__":
    sdk_init()
    sdk_server()



