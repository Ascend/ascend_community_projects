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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def parse_args():
    parser = argparse.ArgumentParser(description='SSIM & PSNR evaluation')
    parser.add_argument('--gpu_results_dir', type=str, default='results/gpu')
    parser.add_argument('--npu_results_dir', type=str, default='results/npu')
    return parser.parse_args()


def eval_metrics(gpu_path, npu_path):
    total_ssim_score = 0
    total_psnr_socre = 0
    pairs = 0

    # Becaure the filename saved by plugin is decided by its timestamp,
    # should ensure the files'order keeping the same as before.
    gpu_imgs = sorted(glob.glob(os.path.join(gpu_path, '*.jpg')))
    npu_imgs = sorted(glob.glob(os.path.join(npu_path, '*.jpg')))
    assert len(gpu_imgs) == len(npu_imgs)

    for index, img1_path in enumerate(gpu_imgs):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(npu_imgs[index])

        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        h = min(img1_h, img2_h)
        w = min(img1_w, img2_w)

        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        ssim_score = ssim(img1, img2, channel_axis=2)
        psnr_score = psnr(img1, img2)
        total_ssim_score += ssim_score
        total_psnr_socre += psnr_score
        pairs += 1
        print('ssim', ssim_score)
        print('psnr', psnr_score)

    print('avg ssim', total_ssim_score / pairs)
    print('avg psnr', total_psnr_socre / pairs)


if __name__ == "__main__":
    args = parse_args()
    eval_metrics(args.gpu_results_dir, args.npu_results_dir)
