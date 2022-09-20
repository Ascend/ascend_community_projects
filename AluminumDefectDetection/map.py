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
import glob
import sys
import os
import numpy as np


def error(msg):
    """
    throw error and exit
    """
    print(msg)
    sys.exit(0)


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    a = box1[:, None, 2:]
    b = box1[:, None, :2]
    inter = (np.minimum(a, box2[:, 2:]) - np.maximum(b, box2[:, :2]))
    inter = inter.clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = np.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(np.int)] = matches[:, 2:3] >= iouv
    return correct


def file_lines_to_list(path):
    """
    Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def parse_line(txt_file, lines_list, bounding_boxes, already_seen_classes):
    """ parse line
        :param txt_file:
        :param lines_list:
        :param bounding_boxes:
        :param counter_per_class:
        :param already_seen_classes:
        :return: bounding_boxes, counter_per_class
    """
    for line in lines_list:
        try:
            if line == '':
                continue
            class_name, left, top, right, bottom = line.split()
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <l> <t> <r> <b>\n"
            error_msg += " Received: " + line
            error(error_msg)
        bbox = left + " " + top + " " + right + " " + bottom
        bbox_list = [float(left), float(top), float(right), float(bottom)]
        bounding_boxes.append({"class_name": int(class_name), "bbox": bbox_list, "used": False})

        if class_name not in already_seen_classes:
            already_seen_classes.append(class_name)
    return bounding_boxes


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


dict_classes = {
    "non_conduct": "0", "abrasion_mark": "1", "corner_leak": "2", "orange_peel": "3", "leak": "4", "jet_flow": "5",
    "paint_bubble": "6", "pit": "7", "motley": "8", "dirty_spot": "9"
}




def load_pred(path):
    class_list = file_lines_to_list(path)
    if len(class_list):
        pred = np.zeros((len(class_list), 6))
    else:
        return np.zeros((0, 6))
    for index, line in enumerate(class_list):
        sl = line.split()
        tmp_class_name, left, top, right, bottom, confidence = sl
        pred[index, 0:4] = float(left), float(top), float(right), float(bottom)
        pred[index, 4] = float(confidence)
        pred[index, 5] = int(tmp_class_name)
    return pred


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap


def map_for_all_classes(tp, conf, pred_cls, target_cls, eps=1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    ap = np.zeros((nc, tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return ap


names = {
    0: "non_conduct", 1: "abrasion_mark", 2: "corner_leak", 3: "orange_peel", 4: "leak", 5: "jet_flow",
    6: "paint_bubble", 7: "pit", 8: "motley", 9: "dirty_spot"
}


def map_cac(opt):
    file_path = opt.gt
    pre_file_path = opt.test_path

    files_list = glob.glob(file_path + '/*.txt')
    if len(files_list) == 0:
        error("Error: No ground-truth files found!")
    files_list.sort()
    # dictionary with counter per class
    file_bbox = {}
    iouv = np.linspace(0.5, 0.95, 10)
    niou = 10
    stats = []
    for txt_file in files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(file_path, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        already_seen_classes = []
        boxes = parse_line(txt_file, lines_list, bounding_boxes,
                           already_seen_classes)

        predn = load_pred(pre_file_path + file_id + ".txt")

        predn[:, 0:4] = xywh2xyxy(predn[:, 0:4])
        predn[:, [0, 2]] *= 2560
        predn[:, [1, 3]] *= 1920
        if len(boxes):
            labelsn = np.zeros((len(boxes), 5))
            for index, item in enumerate(boxes):
                labelsn[index, 1:] = item['bbox']
                labelsn[index, 0:1] = item['class_name']

            labelsn[:, 1:] = xywh2xyxy(labelsn[:, 1:])
            labelsn[:, [1, 3]] *= 2560
            labelsn[:, [2, 4]] *= 1920
            tcls = labelsn[:, 0]

            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = np.zeros(predn.shape[0], niou, dtype=bool)
        stats.append((correct, predn[:, 4], predn[:, 5], tcls))
        print(boxes)

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    ap = map_for_all_classes(*stats)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    print(f"mAP0.5:  {ap50.mean()}")
    print(f"mAP0.5:0.95:  {ap.mean()}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default="./test/gt/", help='')
    parser.add_argument('--test_path', type=str, default="./test/test_out_txt/", help='')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    map_cac(parse_opt())
