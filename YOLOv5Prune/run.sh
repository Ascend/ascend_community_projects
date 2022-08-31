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

if [ $# -le 1 ]
then
    echo "Usage:"
    echo "bash run.sh [task_type][image_set][image_dir] or bash run.sh eval [dataset_path]"
    exit 1
fi
get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
if [[ "${1}"x = "eval"x ]]
then
    task_type=$1
    dataset_path=$(get_real_path $2)
    image_set=$(get_real_path $2)"/VOC2007/ImageSets/Main/test.txt"
    image_dir=$(get_real_path $2)"/VOC2007/JPEGImages"
    if [[ ! -d $dataset_path ]]
    then 
        echo "error : $dataset_path is not a dir"
        exit 1
    fi
elif [[ "${1}"x = "detect"x || "${1}"x = "speed"x ]]
then
    task_type=$1
    image_set=$(get_real_path $2)
    image_dir=$(get_real_path $3)
else 
    echo "Undefined task!"
    exit 1
fi
if [[ ! -f $image_set ]]
then 
    echo "error : $image_set is not a file"
    exit 1
fi

if [[ ! -d $image_dir ]]
then 
    echo "error : $image_dir is not a dir"
    exit 1
fi

file=main
if [[ ! -f $file ]]
then
    bash build.sh
fi

# run
echo "start" $task_type "task"
echo "image_set" $image_set 
echo "image_dir" $image_dir
./main $task_type $image_set $image_dir
if [[ "$task_type"x = "eval"x ]]
then 
    echo "compute mAP..."
    python compute_mAP/reval_voc.py txt_result --voc_dir $dataset_path
fi
exit 0