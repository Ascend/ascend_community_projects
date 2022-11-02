from math import sqrt
from itertools import product
import numpy as np


def get_anchors(input_shape, anchors_size):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    
    all_anchors = []
    for i, _ in enumerate(feature_heights):
        anchors     = make_anchors(feature_heights[i], feature_widths[i], anchors_size[i], input_shape, [1, 1 / 2, 2])
        all_anchors += anchors
    
    all_anchors = np.reshape(all_anchors, [-1, 4])
    return all_anchors


def get_img_output_length(height, width):
    filter_sizes    = [7, 3, 3, 3, 3, 3, 3]
    padding         = [3, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths  = []

    for i, _ in enumerate(filter_sizes):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]


def make_anchors(conv_h, conv_w, scale, input_shape, aspect_ratios):
    prior_data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / input_shape[1]
            h = scale / ar / input_shape[0]

            prior_data += [x, y, w, h]

    return prior_data