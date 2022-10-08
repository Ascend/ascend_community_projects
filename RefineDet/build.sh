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

path_cur="$(dirname "$0")"

function build_refinedet()
{
    cd "$path_cur" || exit
    rm -rf build
    mkdir -p build
    cd build || exit
    cmake ..
    make
    ret=$?
    if [ ${ret} -ne 0 ]; then
        echo "Failed to build refinedet."
        exit ${ret}
    fi
}

build_refinedet