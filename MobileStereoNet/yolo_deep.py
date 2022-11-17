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

import math
import collections
import cv2
import numpy as np
import camera_configs
from yolov3_infer import yolo_infer

LEFTIMG_PATH = "./image/left_0.jpg"
RIGHTIMG_PATH = "./image/right_0.jpg"
YOLO_RESIZELEN = 416


def get_rectify(height, width):
    left_matrix = camera_configs.left_camera_matrix
    right_matrix = camera_configs.right_camera_matrix
    left_distortion = camera_configs.left_distortion
    right_distortion = camera_configs.right_distortion
    R = camera_configs.R
    T = camera_configs.T

    # 图像尺寸
    size = (width, height)

    # 进行立体更正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_matrix, left_distortion,
                                                      right_matrix, right_distortion, size, R, T)
    # 计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
    
    Camera = collections.namedtuple('Camera', ['left_map1', 'left_map2', 'right_map1', 'right_map2', 'Q'])

    camera = Camera(left_map1, left_map2, right_map1, right_map2, Q)
    return camera


def stereo_match(imgleft, imgright):
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=16 * 6,
                                   blockSize=5,
                                   P1=216,
                                   P2=864,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=0,
                                   speckleRange=1,
                                   preFilterCap=60,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(imgleft, imgright)
    return disparity


if __name__ == '__main__':
    img1 = cv2.imread(LEFTIMG_PATH)
    img2 = cv2.imread(RIGHTIMG_PATH)
    
    img_height, img_width = img1.shape[0:2]

    configs = get_rectify(img_height, img_width)

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(img1, configs.left_map1, configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, configs.right_map1, configs.right_map2, cv2.INTER_LINEAR)
    cv2.imwrite("SGBM_left.jpg", img1_rectified)

    # 将图片置为灰度图，为StereoSGBM作准备 
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 根据SGBM/Semi-Global Block Matching方法生成差异图
    left_match = stereo_match(imgL, imgR)

    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(left_match.astype(np.float32) / 16., configs.Q)

    # 因为om模型读取要YUV格式，前面cv读取处理是BGR，我暂时没找到直接定义Image类的方法，所以重读一遍重构后的图片
    coordinate = yolo_infer("SGBM_left.jpg", YOLO_RESIZELEN)

    for coor in coordinate:
        x = coor.x1 
        y = coor.y1  

        x = int(x)
        y = int(y)

        print('\n像素坐标 x = %d, y = %d' % (x, y))

        x = x - 1
        y = y - 1
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")
        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")
