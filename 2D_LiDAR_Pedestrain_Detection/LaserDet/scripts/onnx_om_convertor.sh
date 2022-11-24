#!/bin/bash
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

atc --model=$1/drow3_drow_e40.onnx --framework=5 --output=drow3_drow_e40 -soc_version=Ascend310
atc --model=$1/dr_spaam_drow_e40.onnx --framework=5 --output=dr_spaam_drow_e40 -soc_version=Ascend310
atc --model=$1/drow3_jrdb_e40.onnx --framework=5 --output=drow3_jrdb_e40 -soc_version=Ascend310
atc --model=$1/dr_spaam_jrdb_e20.onnx --framework=5 --output=dr_spaam_jrdb_e20 -soc_version=Ascend310