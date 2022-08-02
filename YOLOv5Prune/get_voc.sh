
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

start=$(date +%s)
mkdir -p ../tmp
cd ../tmp/

# Download/unzip images and labels
d='.' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f1=VOCtrainval_06-Nov-2007.zip # 446MB, 5012 images
f2=VOCtest_06-Nov-2007.zip     # 438MB, 4953 images
for f in $f2 $f1; do
  echo 'Downloading' $url$f '...' 
  curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
done
wait # finish background tasks

end=$(date +%s)
runtime=$((end - start))
echo "Completed in" $runtime "seconds"