#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from test_pkg.srv import MyResult, MyResultRequest
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
import base64

def saveInferResultToImage(inferResultString, imagePath):

    img = cv2.imread(imagePath)
    rows = img.shape[0]
    cols = img.shape[1]
    retDict = json.loads(inferResultString)
    idsLen = len(retDict["MxpiObject"])
    for i in range(idsLen):

        x0 = int(retDict["MxpiObject"][i]["x0"])
        y0 = int(retDict["MxpiObject"][i]["y0"])
        x1 = int(retDict["MxpiObject"][i]["x1"])
        y1 = int(retDict["MxpiObject"][i]["y1"])
        
        className = retDict["MxpiObject"][i]["classVec"][0]["className"]
        confidence = retDict["MxpiObject"][i]["classVec"][0]["confidence"]

        thickness = int((rows + cols) / 2 * 0.005 + 1)
        wordSize = thickness * 0.3 if thickness * 0.3 > 1.0 else 1.0
        bgr = np.random.randint(0, 255, 3, dtype = np.int32)#随机颜色
        cv2.rectangle(img, (x0, y0), (x1, y1), (int(bgr[0]), int(bgr[1]), int(bgr[2])), thickness)
        cv2.putText(img, className, (x0, y0 - thickness), cv2.FONT_HERSHEY_SIMPLEX, wordSize, (int(bgr[0]), int(bgr[1]), int(bgr[2])), thickness)

    savePath = "/home/HwHiAiUser/hhy/pic/test1__save.jpg"
    cv2.imwrite(savePath, img)

def sdk_client_func():
	# ROS节点初始化
    rospy.init_node('sdk_client')

	# 发现/spawn服务后，创建一个服务客户端，连接名为/spawn的service
    rospy.wait_for_service('/sdk_infer')
    try:
        sdk_client = rospy.ServiceProxy('/sdk_infer', MyResult)
        imagestr = ""
        realimagestr = ""
        imagePath = "/home/HwHiAiUser/hhy/pic/test1.jpg"
        with open(imagePath, 'rb') as f:
            imagebytes = base64.b64encode(f.read())
            f.close()    
            imagestr = str(imagebytes)
            realimagestr = imagestr[2:len(imagestr)-1]

		# 请求服务调用，输入请求数据
        response = sdk_client(realimagestr)

        inferResultString = response.inferResultStr
        saveInferResultToImage(inferResultString, imagePath)
        return inferResultString
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
	#服务调用并显示调用结果
    print("Show response result : \n %s" %(sdk_client_func()))


