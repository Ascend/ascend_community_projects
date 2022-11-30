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

path_cur="$(dirname "$0")"
mkdir include
mkdir evaluate
cp $path_cur/Pytorch_Retinaface/layers/functions/prior_box.py $path_cur/include
cp $path_cur/Pytorch_Retinaface/utils/box_utils.py $path_cur/include
cp $path_cur/Pytorch_Retinaface/utils/nms/py_cpu_nms.py $path_cur/include
cp -R $path_cur/Pytorch_Retinaface/widerface_evaluate $path_cur/evaluate
cp $path_cur/widerface/val/wider_val.txt $path_cur/evaluate