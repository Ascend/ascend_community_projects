# ------------------------------------------------------------------------------
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

import math
import numpy as np
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


def visualize(frame, result, dataset='coco'):
    '''
    frame: frame image
    result: result of predictions
    dataset: coco or mpii

    return rendered image
    '''
    if dataset == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191),
                   (127, 77, 255), (77, 255, 127), (0, 255, 255)]
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif dataset == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    img = frame
    for human in result:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = np.concatenate((kp_preds, np.expand_dims((kp_preds[5, :]+kp_preds[6, :]) / 2.0, 0)))
        kp_scores = np.concatenate((kp_scores, np.expand_dims((kp_scores[5, :]+kp_scores[6, :]) / 2.0, 0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
            transparency = float(max(0, min(1, kp_scores[n])))
            img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
        for i, (start_pair, end_pair) in enumerate(l_pair):
            if start_pair in part_line and end_pair in part_line:
                start_xy = part_line.get(start_pair)
                end_xy = part_line.get(end_pair)
                bg = img.copy()

                coord_x = (start_xy[0], end_xy[0])
                coord_y = (start_xy[1], end_xy[1])
                coord_mx = np.mean(coord_x)
                coord_my = np.mean(coord_y)
                length = ((coord_y[0] - coord_y[1]) ** 2 + (coord_x[0] - coord_x[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(coord_y[0] - coord_y[1], coord_x[0] - coord_x[1]))
                stickwidth = (kp_scores[start_pair] + kp_scores[end_pair]) + 1
                polygon = cv2.ellipse2Poly((int(coord_mx), int(coord_my)), (int(length/2),
                                           int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):
                    cv2.fillConvexPoly(bg, polygon, line_color[i])
                else:
                    cv2.line(bg, start_xy, end_xy, (255, 255, 255), 1)
                if n < len(p_color):
                    transparency = float(max(0, min(1, 0.5 * (kp_scores[start_pair] + kp_scores[end_pair])-0.1)))
                else:
                    transparency = float(max(0, min(1, (kp_scores[start_pair] + kp_scores[end_pair]))))

                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    return img
