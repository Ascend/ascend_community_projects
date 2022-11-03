from math import sqrt
from itertools import product
import numpy as np


def get_anchors(shape, size):
    heights, widths = output_length(shape[0], shape[1])
    
    all = []
    for i, _ in enumerate(heights):
        anchors     = make(heights[i], widths[i], size[i], shape, [1, 1 / 2, 2])
        all += anchors
    
    all = np.reshape(all, [-1, 4])
    return all


def output_length(height, width):
    sizes    = [7, 3, 3, 3, 3, 3, 3]
    ding         = [3, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2]
    heights = []
    widths  = []

    for i, _ in enumerate(sizes):
        height  = (height + 2*ding[i] - sizes[i]) // stride[i] + 1
        width   = (width + 2*ding[i] - sizes[i]) // stride[i] + 1
        heights.append(height)
        widths.append(width)
    return np.array(heights)[-5:], np.array(widths)[-5:]


def make(conv_h, conv_w, s, shape, aspect_ratios):
    data = []
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in aspect_ratios:
            ar = sqrt(ar)
            w = s * ar / shape[1]
            h = s / ar / shape[0]

            data += [x, y, w, h]

    return data