"""Reval = re-eval. Re-evaluate saved detections."""

import os
import sys
import argparse
import numpy as np
import _pickle as cPickle

from voc_eval import voc_eval


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--voc_dir', dest='voc_dir', default='data/VOCdevkit', type=str)
    parser.add_argument('--year', dest='year', default='2007', type=str)
    parser.add_argument('--image_set', dest='image_set', default='test', type=str)

    parser.add_argument('--classes', dest='class_file', default='models/yolov5/voc.names', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def get_voc_results_file_template(image_set, out_dir = '.'):
    filename = 'det_' + image_set + '_{:s}.txt'
    path = os.path.join(out_dir, filename)
    return path


def do_python_eval(devkit_path, year, image_set, classes, output_dir):
    annopath = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
        '{}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    use_07_metric = False

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = get_voc_results_file_template(image_set, output_dir).format(cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.55,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        os.system('touch '+os.path.join(output_dir, cls + '_pr.pkl'))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'rb+') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')



if __name__ == '__main__':
    args = parse_args()

    res_dir = os.path.abspath(args.output_dir[0])
    print(output_dir)
    with open(args.class_file, 'r') as file:
        lines = file.readlines()

    class_name = [t.strip('\n') for t in lines]
    print('Evaluating detections')
    do_python_eval(args.voc_dir, args.year, args.image_set, class_name, res_dir)