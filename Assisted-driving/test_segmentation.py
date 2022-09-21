# -*-coding:utf-8-*-

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

import os
import argparse
import numpy as np
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GROUND_TRUTH_MASK_FOLDER",
        type=str,
        default="./data/road_line/data/validation/v1.2/labels",
        help="path to dataset")
    parser.add_argument(
        "--PREDICT_MASK_FOLDER",
        type=str,
        default="./data/road_line/pre_mask",
        help="path to dataset")
    opt = parser.parse_args()

    gt_masks = [os.path.join(opt.GROUND_TRUTH_MASK_FOLDER, i)
                for i in os.listdir(opt.GROUND_TRUTH_MASK_FOLDER)]
    IOUs = []
    for i, _ in enumerate(gt_masks[:]):
        pr_mask_file = os.path.join(
            opt.PREDICT_MASK_FOLDER, _.split('/')[-1].replace('.png', '.jpg'))

        if os.path.isfile(pr_mask_file) and os.path.isfile(gt_masks[i]):
            imgPredict = cv2.imread(pr_mask_file, 0)
            imgLabel = cv2.imread(gt_masks[i], 0)
            _, imgPredict = cv2.threshold(
                imgPredict, 240, 1, cv2.THRESH_BINARY)
            _, imgLabel = cv2.threshold(imgLabel, 240, 1, cv2.THRESH_BINARY)
            dst_and = cv2.bitwise_and(imgLabel, imgPredict)
            dst_or = cv2.bitwise_or(imgLabel, imgPredict)
            dst_and = np.sum(dst_and)
            dst_or = np.sum(dst_or)
            if dst_and < 20 and dst_or == 0:
                continue
            IoU = dst_and / dst_or
            IOUs.append(IoU)
        else:
            pass
    print('mIoU is : ', np.mean(IOUs))
