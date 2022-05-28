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
import rospy
from test_pkg.srv import MyResult, MyResultRequest


def save_infer_result_to_image(infer_result_string, image_path):

    img = cv2.imread(image_path)
    rows = img.shape[0]
    cols = img.shape[1]
    ret_dict = json.loads(infer_result_string)
    ids_len = len(ret_dict["MxpiObject"])
    for i in range(ids_len):

        x0 = int(ret_dict["MxpiObject"][i]["x0"])
        y0 = int(ret_dict["MxpiObject"][i]["y0"])
        x1 = int(ret_dict["MxpiObject"][i]["x1"])
        y1 = int(ret_dict["MxpiObject"][i]["y1"])
        
        class_name = ret_dict["MxpiObject"][i]["classVec"][0]["className"]
        confidence = ret_dict["MxpiObject"][i]["classVec"][0]["confidence"]

        thickness = int((rows + cols) / 2 * 0.005 + 1)
        word_size = thickness * 0.3 if thickness * 0.3 > 1.0 else 1.0
        bgr = np.random.randint(0, 255, 3, dtype = np.int32)
        color_random = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        text_pos = (x0, y0 - thickness)
        text_form = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img, (x0, y0), (x1, y1), (int(bgr[0]), int(bgr[1]), int(bgr[2])), thickness)
        cv2.putText(img, class_name, text_pos, text_form, word_size, color_random, thickness)

    save_path = "/home/HwHiAiUser/pic/test1__save.jpg"
    cv2.imwrite(save_path, img)


def sdk_client_func():
	# ROS节点初始化
    rospy.init_node('sdk_client')

	# 发现/spawn服务后，创建一个服务客户端，连接名为/spawn的service
    rospy.wait_for_service('/sdk_infer')
    try:
        sdk_client = rospy.ServiceProxy('/sdk_infer', MyResult)
        imagestr = ""
        realimagestr = ""
        image_path = "/home/HwHiAiUser/pic/test1.jpg"
        with open(image_path, 'rb') as f:
            imagebytes = base64.b64encode(f.read())
            f.close()    
            imagestr = str(imagebytes)
            realimagestr = imagestr[2:len(imagestr)-1]

		# 请求服务调用，输入请求数据
        response = sdk_client(realimagestr)

        infer_result_string = response.inferResultStr
        save_infer_result_to_image(infer_result_string, image_path)
        return infer_result_string
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return ""

if __name__ == "__main__":
	#服务调用并显示调用结果
    print("Show response result : \n %s" % (sdk_client_func()))

