
import sys
import re
import json
import os
import cv2
import random
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
import signal
import datetime
import threading


def sigint_handler(signum, frame):
    global is_sigint_up
    is_sigint_up = True
    print("catched interrupt signal")

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
is_sigint_up = False

if __name__ == '__main__':

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    path = b"./pipeline/multi_infer2_4.pipeline"
    ret = streamManagerApi.CreateMultipleStreamsFromFile(path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    streamName = b'inferofflinevideo'
    count = 0
    # beginTime = datetime.datetime.now()
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
            if is_sigint_up:
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
        if is_sigint_up:
            print("Exit")
            break    
        retStr = inferResult.data.decode()
        print(retStr)
        
    streamManagerApi.DestroyAllStreams()