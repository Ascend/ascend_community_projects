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

import glob
import os
import argparse
import cv2
import numpy as np

# thresholds for accuracy
THRESHOLD_ONE = 1.25
THRESHOLD_TWO = 1.25 ** 2
THRESHOLD_TRHEE = 1.25 ** 3


def get_args():
    parser = argparse.ArgumentParser(description='Postprocess.')
    parser.add_argument('--gt_path', type=str, default="dataset/depth_info",
                        help='path of the folder which contains groundtruth files.')
    parser.add_argument('--pred_path', type=str, default="results",
                        help='path of the folder which contains pred files.')

    return parser.parse_args()


def bilinear_sampling(source, destination_width, destination_height):
    """
    Bilinear sampling of source data

        |   upper left           |
     y2 |-------P1-----|---------P2---
        |       |      |         |
        |       |      |         |
      y |--------------P-------------
        |       |      |         |
        |       |      |         |
     y1 |-------P3---------------P4---
        |       |      |         | lower right
        |       |      |         |
        |----------------------------
               x1      x        x2
    f(x,y) = (1 - w1) * (1 - w2) * value(P1) +
             (1 - w1) * w2 * value(P2) +
             w1 * (1 - w2) * value(P3) +
             w1 * w2 * value(P4)

    :param source: source data
    :param destination_height: the height of destination data
    :param destination_width: the width of destination data
    :return: output data after bilinear sampling
    """
    # source data size
    src_capacity = source.shape[0]
    src_height = source.shape[1]
    src_width = source.shape[2]
    # destination data size
    dst_height = destination_height
    dst_width = destination_width

    # scale factor
    scale_height = src_height / dst_height
    scale_width = src_width / dst_width

    # calculate the corresponding coordinate of source data
    x_index = np.array([x for x in range(int(dst_width))])
    y_index = np.array([y for y in range(int(dst_height))])
    src_x = (x_index + 0.5) * scale_width - 0.5
    src_y = (y_index + 0.5) * scale_height - 0.5
    src_x = np.repeat(np.expand_dims(src_x, axis=0), dst_height, axis=0)
    src_y = np.repeat(np.expand_dims(src_y, axis=1), dst_width, axis=1)

    # rounded down, get the row and column number of upper left corner
    src_x_int = np.floor(src_x)
    src_y_int = np.floor(src_y)
    # take out the decimal part to construct the weight
    src_x_float = src_x - src_x_int
    src_y_float = src_y - src_y_int
    # expand to input data size
    src_x_float = np.repeat(np.expand_dims(
        src_x_float, axis=0), src_capacity, axis=0)
    src_y_float = np.repeat(np.expand_dims(
        src_y_float, axis=0), src_capacity, axis=0)

    # get upper left and lower right index
    left_x_index = src_x_int.astype(int)
    upper_y_index = src_y_int.astype(int)
    right_x_index = left_x_index + 1
    lower_y_index = upper_y_index + 1

    # boundary condition
    left_x_index[left_x_index < 0] = 0
    upper_y_index[upper_y_index < 0] = 0
    right_x_index[right_x_index > src_width - 1] = src_width - 1
    lower_y_index[lower_y_index > src_height - 1] = src_height - 1

    # upper left corner data
    upper_left_value = source[:, upper_y_index, left_x_index]
    # upper right corner data
    upper_right_value = source[:, upper_y_index, right_x_index]
    # lower left corner data
    lower_left_value = source[:, lower_y_index, left_x_index]
    # lower right corner data
    lower_right_value = source[:, lower_y_index, right_x_index]

    # bilinear sample
    target = (1. - src_y_float) * (1. - src_x_float) * upper_left_value + \
             (1. - src_y_float) * src_x_float * upper_right_value + \
        src_y_float * (1. - src_x_float) * lower_left_value + \
        src_y_float * src_x_float * lower_right_value

    return target.squeeze()


def calculate_error(ground_truth, predict):
    """
    calculate errors between ground truth and predict value
    :param ground_truth: ground truth
    :param predict: predict value
    :return: errors (Absolute relative error, Square relative error, 
                     Root mean square error, Log root mean square error, log 10 error and accuracy)
    """
    # calculate absolute relative error
    abs_rel_error = np.abs(ground_truth - predict) / ground_truth
    abs_rel_error = np.mean(abs_rel_error)

    # calculate square relative error
    sq_rel_error = ((ground_truth - predict) ** 2) / ground_truth
    sq_rel_error = np.mean(sq_rel_error)

    # calculate root mean square error
    rmse_error = (ground_truth - predict) ** 2
    rmse_error = np.sqrt(rmse_error.mean())

    # calculate log root mean square error
    log_rmse_error = (np.log(ground_truth) - np.log(predict)) ** 2
    log_rmse_error = np.sqrt(log_rmse_error.mean())

    # calculate log 10 error
    log_10_error = np.abs(np.log10(ground_truth) - np.log10(predict))
    log_10_error = log_10_error.mean()

    # calculate accuracy
    thresh = np.maximum((ground_truth / predict), (predict / ground_truth))
    a1 = (thresh < THRESHOLD_ONE).mean()
    a2 = (thresh < THRESHOLD_TWO).mean()
    a3 = (thresh < THRESHOLD_TRHEE).mean()

    return dict(abs_rel_error=abs_rel_error, sq_rel_error=sq_rel_error, rmse_error=rmse_error,
                log_rmse_error=log_rmse_error, log_10_error=log_10_error, accuracy=[a1, a2, a3])


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.gt_path):
        print("groundtruth path does not exist,please check it.")
        exit(-1)
    if not os.path.exists(args.pred_path):
        print("pred files path does not exist,please check it.")
        exit(-1)

    print("start to collect groundtruth files……")
    depth_true = []
    for file in sorted(glob.glob(os.path.join(args.gt_path, '*.npy'))):
        depth_true.append(np.load(file))
    depth_true = np.asarray(depth_true)

    if len(depth_true) == 0:
        print("there is no valid npy file in groundtruth path,please check it.")
        exit(-1)

    height, width = depth_true.shape[-2:]

    print("start to collect pred files……")
    depth_pred = []
    for file in sorted(glob.glob(os.path.join(args.pred_path, "*.tiff"))):
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = np.expand_dims(image, 0)
        image = bilinear_sampling(image, width, height)
        depth_pred.append(image)
    depth_pred = np.asarray(depth_pred)

    if len(depth_pred) != len(depth_true):
        print("nums of pred files and groundtruth don't match,please check it.")
        exit(-1)

    print("start to calculate accuarcy……")
    errors = calculate_error(depth_true, depth_pred)

    # print evaluation result
    image_num = depth_pred.shape[0]
    print('absolute relative error on {} test images is {}'.format(
        image_num, errors.get('abs_rel_error', -1)))
    print('square relative error on {} test images is {}'.format(
        image_num, errors.get('sq_rel_error', -1)))
    print('root mean square error on {} test images is {}'.format(
        image_num, errors.get('rmse_error', -1)))
    print('log root mean square error on {} test images is {}'.format(
        image_num, errors.get('log_rmse_error', -1)))
    print('log 10 error on {} test images is {}'.format(
        image_num, errors.get('log_10_error', -1)))
    print('accuracy on {} test images is {}'.format(
        image_num, errors.get('accuracy', -1)))
