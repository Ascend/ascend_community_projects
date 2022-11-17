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

import re
import xml.dom.minidom as xml
import cv2
import numpy as np

dom = xml.parse('camera.xml')
root = dom.documentElement
params = root.getElementsByTagName('data')
camera_list = []
for matrix in params:
    line = re.split(r"\s|\n", matrix.firstChild.data)
    while "" in line:
        line.remove("")
    camera_list.append(list(map(float, line)))

left_camera_matrix = np.array([camera_list[0][:3],
                               camera_list[0][3:6],
                               camera_list[0][6:]])
left_distortion = np.array([camera_list[1]])

right_camera_matrix = np.array([camera_list[2][:3],
                                camera_list[2][3:6],
                                camera_list[2][6:]])
right_distortion = np.array([camera_list[3]])

# 旋转关系向量
R = np.array([camera_list[4][:3],
              camera_list[4][3:6],
              camera_list[4][6:]])

# 平移关系向量
T = np.array(camera_list[5])
