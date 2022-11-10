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

set -e

# 该脚本用来将onnx模型文件转换成.om模型文件
# This is used to convert onnx model file to .om model file.


# 设置环境变量（请确认install_path路径是否正确以及包含Python路径的PATH是否正确）
# Set environment PATH (Please confirm that the install_path is correct).

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp


# 执行，转换Adabins_nyu模型
# Execute, transform Adabins_nyu model.

atc --model=./AdaBins_nyu.onnx --framework=5 --output=./AdaBins_nyu --soc_version=Ascend310 --insert_op_conf=./aipp_adabins_640_480.aippconfig --log=error

# 退出
# exit
exit 0