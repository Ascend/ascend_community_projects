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
import random
import signal
import datetime
import threading
import cv2
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


def sigint_handler(signum, frame):
    signum = signum
    frame = frame
    global ISSIGINTUP
    ISSIGINTUP = True
    print("catched interrupt signal")

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
ISSIGINTUP = False

if __name__ == '__main__':

    multiStreamManagerApi = StreamManagerApi()
    ret = multiStreamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, retStr=%s" % str(ret))
        multiStreamManagerApi.DestroyAllStreams()
        exit()

    PIPELINE_PATH = b"./pipeline/multi_infer1_8.pipeline"
    ret = multiStreamManagerApi.CreateMultipleStreamsFromFile(PIPELINE_PATH)
    if ret != 0:
        print("Failed to create Stream, retStr=%s" % str(ret))
        multiStreamManagerApi.DestroyAllStreams()
        exit()

    STREAM_NAME = b'inferofflinevideo'
    COUNT = 0

    def time_func():
        time_step = 0
        time_count = 0
        begin_time = datetime.datetime.now()
        one_step = 2
        
        while True:
            cur_time = (datetime.datetime.now() - begin_time).total_seconds()
            if cur_time >= (time_step + one_step):
                time_step = time_step + one_step
                print("rate:", (COUNT - time_count) * 1.0 / one_step)
                time_count = COUNT
            if ISSIGINTUP:
                print("Exit")
                break 

    t = threading.Thread(target=time_func, args=())
    t.start()
    while True:
        if ISSIGINTUP:
            print("Exit")
            break
        
        inferResult = multiStreamManagerApi.GetResult(STREAM_NAME, 0, 10000)
        COUNT = COUNT + 1

        if inferResult is None:
            continue
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            continue
 
        retStr = inferResult.data.decode()
        print(retStr)
        
    multiStreamManagerApi.DestroyAllStreams()