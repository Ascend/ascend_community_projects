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
import h5py
import imageio
import numpy as np
from tqdm import trange

if __name__ == '__main__':

    DATA_FILE_PATH = './dataset/nyu_depth_v2_labeled.mat'
    OUTPUT_IMAGE_PATH = './dataset/full_set'
    OUTPUT_DEPTH_INFO_PATH = './dataset/depth_info'

    # check origin data file
    if os.path.exists(DATA_FILE_PATH) != 1:
        print('The {} does not exist, please check first.'.format(DATA_FILE_PATH))
        exit(1)

    if os.path.exists(OUTPUT_IMAGE_PATH) != 1 or os.path.exists(OUTPUT_DEPTH_INFO_PATH) != 1:
        print('The {} or {} does not exist, please check first.'.format(
            OUTPUT_IMAGE_PATH, OUTPUT_DEPTH_INFO_PATH))
        exit(1)

    # load data file
    data = h5py.File(DATA_FILE_PATH)

    # get image data
    image = np.array(data['images'])
    image = np.transpose(image, (0, 2, 3, 1))

    # get depth info data
    depth = np.array(data['depths'])

    # save image
    for i in trange(image.shape[0]):
        out_img = image[i, :, :, :]
        out_img = out_img.transpose(1, 0, 2)
        imageio.imwrite(OUTPUT_IMAGE_PATH + '/' + str(i) + '.jpg', out_img)

    # save depth info
    for i in trange(depth.shape[0]):
        out_depth = depth[i, :, :]
        out_depth = out_depth.transpose(1, 0)
        np.save(OUTPUT_DEPTH_INFO_PATH + '/' + str(i) + '.npy', out_depth)
