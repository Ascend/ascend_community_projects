#!/bin/bash

# Copyright 2020 Huawei Technologies Co., Ltd
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

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$PYTHONPATH:${CUR_PATH}/keyword-spotting-main/

cd ./modelchange
python3.9 keras2onnx.py # .h5 to pb
python3.9 -m tf2onnx.convert --saved-model tmp_model --output model1.onnx --opset 11 # pb to onnx
python3.9 onnx2om.py
atc --framework=5 --model=model2.onnx --output=modelctc1 --input_format=ND --input_shape="input:1,958,13" --soc_version=Ascend310

exit 0