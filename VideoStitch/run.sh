#!/bin/bash
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

# 4路视频路径
VIDEO0=""   # 左上角视频路径，作为拼接基准
VIDEO1=""
VIDEO2=""
VIDEO3=""

# 要拼接的帧数，若为0表示拼接完整视频所有帧
FRAMES=0

# 是否保存结果，若为0，则只执行拼接过程不输出视频，若为1则执行拼接与保存视频操作
VIDEO_GLAG=1

# 特征点提取阈值，若视频重叠部分纹理不明显或视频清晰度较差，建议适当减小阈值，阈值范围(0,10000)
MINHESSIAN=2000

./main $FRAMES $VIDEO_GLAG $MINHESSIAN $VIDEO0 $VIDEO1 $VIDEO2 $VIDEO3
echo "Stitch finish!"