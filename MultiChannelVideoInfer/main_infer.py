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


def SigintHandler(signum, frame):
    global isSigintUp
    isSigintUp = True
    print("catched interrupt signal")

signal.signal(signal.SIGINT, SigintHandler)
signal.signal(signal.SIGHUP, SigintHandler)
signal.signal(signal.SIGTERM, SigintHandler)
isSigintUp = False

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
            if isSigintUp:
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
        if isSigintUp:
            print("Exit")
            break    
        retStr = inferResult.data.decode()
        
    streamManagerApi.DestroyAllStreams()