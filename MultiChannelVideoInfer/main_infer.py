#!/usr/bin/env python
# coding=utf-8

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

import sys
import re
import json
import os
import signal
import datetime
import threading
import random
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


def sigint_handler():
    global ISSIGINTUP
    ISSIGINTUP = True
    print("catched interrupt signal")

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
ISSIGINTUP = False

if __name__ == '__main__':

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        streamManagerApi.DestroyAllStreams()
        exit()

    pipelinePath = b"./pipeline/multi_infer1_8.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(pipelinePath)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        streamManagerApi.DestroyAllStreams()
        exit()

    streamName = b'inferofflinevideo'
    count = 0
    def timeFunc():
        timeStep = 0
        timeCount = 0
        beginTime = datetime.datetime.now()
        oneStep = 2
        while True:
            curTime = (datetime.datetime.now() - beginTime).total_seconds()
            if curTime >= (timeStep + oneStep):
                timeStep = timeStep + oneStep
                print("rate:", (count - timeCount) * 1.0 / oneStep)
                timeCount = count
            if ISSIGINTUP:
                print("Exit")
                break 

    t = threading.Thread(target=timeFunc, args=())
    t.start()
    while True:
        inferResult = streamManagerApi.GetResult(streamName, 0, 10000)
        count = count + 1

        if inferResult is None:
            continue
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            continue
        if ISSIGINTUP:
            print("Exit")
            break    
        retStr = inferResult.data.decode()
        
    streamManagerApi.DestroyAllStreams()