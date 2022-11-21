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

import argparse
import numpy as np


# define the arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate AttrRecognition infer result's accuracy")

    parser.add_argument("--pred-file", type=str, default="infer_result.txt",
                        help="infer result file's path.")
    parser.add_argument("--gt-file", type=str, default="test_full.txt",
                        help="ground truth file's path.")

    return parser.parse_args()

if __name__ == '__main__':
    PATH_SHIFT = 0
    ATTR_NUM = 40  # the number of all attributes is 40

    args = parse_args()

    gt_file_path = args.gt_file
    pred_file_path = args.pred_file

    gt_file = open(gt_file_path, 'r')
    pred_file = open(pred_file_path, 'r')

    gt_line = gt_file.readline().strip().split()
    pred_line = pred_file.readline().strip().split()

    equal_nums = np.zeros(ATTR_NUM, dtype=np.int32)
    EVAL_SUM = 0
    while pred_line:
        EVAL_SUM += 1
        for i in range(ATTR_NUM):
            pred_value = int(pred_line[1 + i])
            gt_value = int(gt_line[PATH_SHIFT + 1 + i])
            if pred_value == gt_value:
                equal_nums[i] += 1

        gt_line = gt_file.readline().strip().split()
        pred_line = pred_file.readline().strip().split()

    result = np.zeros(ATTR_NUM)
    INDEX = 0
    for value in equal_nums:
        result[INDEX] = value * 1.0 / EVAL_SUM * 100
        INDEX += 1

    print('mean result', np.mean(result))
