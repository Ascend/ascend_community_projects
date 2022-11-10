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
def get_arguments():
    parser = argparse.ArgumentParser(description="Attribute Network")
    parser.add_argument("--gt-file", type=str,
                        help="ground truth file path.")
    parser.add_argument("--pred-file", type=str,
                        help="prediction file path.")
    parser.add_argument("--path-shift", type=int, default=0,
                        help="prediction file path.")
    return parser.parse_args()


def cal_attr():
    args = get_arguments()
    path_shift = args.path_shift
    print(args)
    gt_attr_file = args.gt_file
    pred_file = args.pred_file
    # the number of the attribute is 40
    attr_num = 40
    gt_f = open(gt_attr_file, 'r')
    gt_line = gt_f.readline().strip().split()
    pred_f = open(pred_file, 'r')
    pred_line = pred_f.readline().strip().split()
    # count the same prediction
    same_count = np.zeros(attr_num, dtype=np.int32)
    valid_sum = 0
    while pred_line:
        valid_sum += 1
        for i in range(attr_num):
            pred_attr = int(pred_line[1 + i])
            gt_attr = int(gt_line[path_shift + 1 + i])
            if pred_attr == gt_attr:
                same_count[i] += 1
        gt_line = gt_f.readline().strip().split()
        pred_line = pred_f.readline().strip().split()
    print(valid_sum)
    result = np.zeros(attr_num)
    cur_index = 0
    for v in same_count:
        # percentage calculation
        print(v * 1.0 / valid_sum * 100)
        result[cur_index] = v * 1.0 / valid_sum * 100
        cur_index += 1
        # calculate the mean result
    print('mean result', np.mean(result))
    return result


if __name__ == '__main__':
    cal_attr()
