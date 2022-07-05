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
set -e

MODE=$1
SPEEDTEST=$2


cd src/

out_path="../out/"
if [ -d "$out_path" ]; then
    rm -rf "$out_path"
else
    echo "file $out_path is not exist."
fi

mkdir -p "$out_path"


if [ ${MODE} = "image" ]; then
	python image.py
elif [ ${MODE} = "video" ]; then
    python video.py ${SPEEDTEST}
elif [ ${MODE} = "evaluate" ]; then
    python evaluate.py
else
    echo -e "The mode must be image or video or evaluate"
fi

exit 0