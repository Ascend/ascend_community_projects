#!/usr/bin/env python
# coding=utf-8

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
import math
import cv2
import numpy as np

YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
PURPLE = (255, 0, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def visualize_fast(frame, result):
    line_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    # Nose, LEye, REye, LEar, REar
    # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
    # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    point_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                (0, 255, 255)]
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    img = frame
    for human in result:
        part_line = {}
        preds = human['keypoints']
        scores = human['keypoints_score']
        # Get the keypoint between the left shoulder and the right shoulder
        lshoulder_index = 5
        rshoulder_index = 6
        middle_preds = np.expand_dims((preds[lshoulder_index, :]+preds[rshoulder_index, :]) / 2.0, 0)
        middle_scores = np.expand_dims((scores[lshoulder_index, :]+scores[rshoulder_index, :]) / 2.0, 0)
        preds = np.concatenate((preds, middle_preds))
        scores = np.concatenate((scores, middle_scores))
        # Draw keypoints
        for n in range(scores.shape[0]):
            if scores[n] <= 0.05:
                continue
            point_x, point_y = int(preds[n, 0]), int(preds[n, 1])
            part_line[n] = (int(point_x), int(point_y))
            if n < len(point_color):
                cv2.circle(img, (point_x, point_y), 3, point_color[n], -1)
            else:
                cv2.circle(img, (point_x, point_y), 1, (255, 255, 255), 2)
        # Draw limbs
        for i, (start_pair, end_pair) in enumerate(line_pair):
            if start_pair in part_line and end_pair in part_line:
                start_point = part_line.get(start_pair)
                end_point = part_line.get(end_pair)
                if i < len(line_color):
                    cv2.line(img, start_point, end_point, line_color[i],
                            2 * int(scores[start_pair] + scores[start_pair]) + 1)
                else:
                    cv2.line(img, start_point, end_point, (255, 255, 255), 1)
    return img


def visualize(frame, result):
    line_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    # Nose, LEye, REye, LEar, REar
    # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
    # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    point_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                (0, 255, 255)]
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    img = frame
    for human in result:
        part_line = {}
        preds = human['keypoints']
        scores = human['keypoints_score']
        # Get the keypoint between the left shoulder and the right shoulder
        lshoulder_index = 5
        rshoulder_index = 6
        middle_preds = np.expand_dims((preds[lshoulder_index, :]+preds[rshoulder_index, :]) / 2.0, 0)
        middle_scores = np.expand_dims((scores[lshoulder_index, :]+scores[rshoulder_index, :]) / 2.0, 0)
        preds = np.concatenate((preds, middle_preds))
        scores = np.concatenate((scores, middle_scores))
        # Draw keypoints
        n = 0
        while n < scores.shape[0]:
            if scores[n] > 0.05:
                point_img = img.copy()
                part_line[n] = (int(preds[n, 0]), int(preds[n, 1]))
                cv2.circle(point_img, (int(preds[n, 0]), int(preds[n, 1])), 2, point_color[n], -1)
                # Now create a mask of logo and create its inverse mask also
                transparency = float(max(0, min(1, scores[n])))
                img = cv2.addWeighted(point_img, transparency, img, 1-transparency, 0)
            n += 1
        # Draw limbs
        for i, (start_pair, end_pair) in enumerate(line_pair):
            if start_pair in part_line and end_pair in part_line:
                line_img = img.copy()
                stick_width = (scores[start_pair] + scores[end_pair]) + 1
                start_point = part_line.get(start_pair)
                end_point = part_line.get(end_pair)

                coord_mx = np.mean((start_point[0], end_point[0]))
                coord_my = np.mean((start_point[1], end_point[1]))

                length = math.sqrt(pow((start_point[1] - end_point[1]), 2) +
                          pow((start_point[0] - end_point[0]), 2))
                angle_degrees = math.atan2(start_point[1] - end_point[1], start_point[0] - end_point[0])
                angle = math.degrees(angle_degrees)
                polygon = cv2.ellipse2Poly((math.floor(coord_mx), math.floor(coord_my)),
                                            (math.floor(length * 0.5), math.floor(stick_width)),
                                            math.floor(angle), 0, 360, 1)
                if i >= len(line_color):
                    cv2.line(line_img, start_point, end_point, (255, 255, 255), 1)
                else:
                    cv2.fillConvexPoly(line_img, polygon, line_color[i])
                if n >= len(point_color):
                    transparency = float(max(0, min(1, (scores[start_pair] + scores[end_pair]))))
                else:
                    transparency = float(max(0, min(1, 0.5 * (scores[start_pair] + scores[end_pair])-0.1)))

                img = cv2.addWeighted(line_img, transparency, img, 1 - transparency, 0)
    return img
