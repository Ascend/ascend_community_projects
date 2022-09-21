#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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
# limitations under the License.mitations under the License.

set -e

CURRENT_PATH="$( cd "$(dirname "$0")" ;pwd -P )"

function build_postprocess_plugin() {
    BUILD_PLUGIN=$CURRENT_PATH/build
    if [ -d "$BUILD_PLUGIN" ]; then
        rm -rf "$BUILD_PLUGIN"
    else
        echo "file $BUILD_PLUGIN is not exist."
    fi
    mkdir -p "$BUILD_PLUGIN"
    cd "$BUILD_PLUGIN"
    cmake ..
    make -j
    cd ..
    exit 0
}

build_postprocess_plugin
exit 0