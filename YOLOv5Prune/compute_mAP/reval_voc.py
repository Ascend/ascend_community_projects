"""
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
"""

import os
import sys
import argparse
import numpy as np
import _pickle as pickle

from voc_eval import evaluate



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--voc_dir', dest='voc_dir', default='data/VOCdevkit', type=str)
    parser.add_argument('--image_set', dest='image_set', default='test', type=str)
    parser.add_argument('--classes', dest='class_file', default='models/yolov5/voc.names', type=str)

    return parser.parse_args()


def get_voc_results_file_template(image_set, out_dir = '.'):
    filename = 'det_' + image_set + '_{:s}.txt'
    path = os.path.join(out_dir, filename)
    return path


def do_eval(devkit_path, image_set, classes, output_dir):
    annopath = os.path.join(
        devkit_path,
        'VOC2007',
        'Annotations',
        '{}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC2007',
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(classes):
        filename = get_voc_results_file_template(image_set, output_dir).format(cls)
        rec, prec, ap = evaluate(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.55)
        aps += [ap]
        
        os.system('touch '+os.path.join(output_dir, cls + '_pr.pkl'))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'rb+') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Results:')
    print('~~~~~~~~')
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    for ap, cls in zip(aps, classes):
        print('AP for {} = {:.4f}'.format(cls, ap))


if __name__ == '__main__':
    args = parse_args()

    res_dir = os.path.abspath(args.output_dir[0])
    with open(args.class_file, 'r') as file:
        lines = file.readlines()
    class_name = [t.strip('\n') for t in lines]
    print('Evaluating detections')
    do_eval(args.voc_dir, args.image_set, class_name, res_dir)