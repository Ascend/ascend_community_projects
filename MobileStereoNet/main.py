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

import os
import glob
import math
import collections
import cv2
import numpy as np
import camera_configs
from yolov3_infer import yolo_infer

LEFTIMG = "left_*.jpg"
RIGHTIMG = "right_*.jpg"
YOLO_RESIZELEN = 416


def get_rectify(height, width):
    left_matrix = camera_configs.left_camera_matrix
    right_matrix = camera_configs.right_camera_matrix
    left_distortion = camera_configs.left_distortion
    right_distortion = camera_configs.right_distortion
    R = camera_configs.R
    T = camera_configs.T

    # Image size
    size = (width, height)

    # Calculate correction transformation
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_matrix, left_distortion,
                                                                      right_matrix, right_distortion, size, R, T)
    # Calculate correction map
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

    left_paths = []
    left_paths.extend(glob.glob(os.path.join("image", LEFTIMG)))
    left_paths.sort()

    right_paths = []
    right_paths.extend(glob.glob(os.path.join("image", RIGHTIMG)))
    right_paths.sort()

    if len(left_paths) == 0 or len(right_paths) == 0:
        print("The dataset is empty!.Please check the dataset and files.")
        exit()

    if len(left_paths) != len(right_paths):
        print("Picture missing!.Please check the dataset and files.")
        exit()

    paths = zip(left_paths, right_paths)

    NUM = 0
    for left, right in paths:
        img1 = cv2.imread(left)
        img2 = cv2.imread(right)

        img_height, img_width = img1.shape[0:2]

        configs = get_rectify(img_height, img_width)

        # Distortion correction
        img1_rectified = cv2.remap(img1, configs.left_map1, configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(img2, configs.right_map1, configs.right_map2, cv2.INTER_LINEAR)
        cv2.imwrite("SGBM_left.jpg", img1_rectified)

        # Set the picture as a grayscale image to prepare for SGBM 
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        # Generate parallax map according to SGBM/Semi-Global Block Matching
        parallax = stereo_match(imgL, imgR)

        # Expand the picture to 3d space, and the value in z direction is the current distance
        threeD = cv2.reprojectImageTo3D(parallax.astype(np.float32) / 16., configs.Q)

        coordinate = yolo_infer("SGBM_left.jpg", YOLO_RESIZELEN)

        NUM += 1
        print("The result of case %d :" % NUM)

        for coor in coordinate:
            x = coor.x1 
            y = coor.y1  
            name = coor.className

            x = int(x)
            y = int(y)

            print('\nPixel coordinates x = {}, y = {}'.format(x, y))

            x = x - 1
            y = y - 1
            print("3D coordinates ({:f}, {:f}, {:f}) mm ".format(threeD[y][x][0], threeD[y][x][1], threeD[y][x][2]))
            distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
            distance = distance / 1000.0  # mm -> m
            print("{}'s actual distance: {:f} m\n".format(name, distance))

    os.remove("SGBM_left.jpg")
