#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
# Description: Set and find the path of the runtime.
# Author: MindX SDK
# Create: 2020
# History: NA

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

export MX_SDK_HOME=/home/wangshengke3/MindX_SDK/mxVision
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64":"/usr/local/Ascend/driver/lib64":${LD_LIBRARY_PATH}
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"

rm -rf ./build
# complie
cmake -S . -Bbuild
make -C ./build  -j

export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":${LD_LIBRARY_PATH}

# run
./main
exit 0