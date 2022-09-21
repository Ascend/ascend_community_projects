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
# limitations under the License.

set -e
CURRENT_PATH="$( cd "$(dirname "$0")" ;pwd -P )"


POSTPROCESS_FOLDER=(
	/postprocess/
)


FLAG=0
for path in ${POSTPROCESS_FOLDER[@]};do
    cd ${CURRENT_PATH}/${path}
    bash build.sh || {
        echo -e "Failed to build postprocess plugin ${path}"
		FLAG=1
    }
done


if [ ${FLAG} -eq 1 ]; then
	exit 1
fi
exit 0
