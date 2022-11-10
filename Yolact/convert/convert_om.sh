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


if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_ONNX_PATH]  [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  [INPUT_ONNX_PATH]  [OUTPUT_OM_PATH_NAME]"

  exit 1
fi

model_path=$1
output_model_name=$2


atc --model=$model_path \
    --framework=5 \
    --output=$output_model_name \
    --input_format=NCHW \
    --op_select_implmode=high_precision \
    --input_shape="images:1,3,544,544" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310

