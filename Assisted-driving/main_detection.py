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

import json
import os
import stat
import glob
import shutil
import argparse
import time
import numpy as np
from PIL import Image, ImageDraw
import cv2
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, InProtobufVector, MxProtobufIn

Class_Label = {
    0: 'i1',
    1: 'i10',
    2: 'i11',
    3: 'i12',
    4: 'i13',
    5: 'i14',
    6: 'i15',
    7: 'i16',
    8: 'i17',
    9: 'i18',
    10: 'i19',
    11: 'i2',
    12: 'i3',
    13: 'i4',
    14: 'i5',
    15: 'i6',
    16: 'i9',
    17: 'il100',
    18: 'il110',
    19: 'il50',
    20: 'il60',
    21: 'il70',
    22: 'il80',
    23: 'il90',
    24: 'ip',
    25: 'ipp',
    26: 'p1',
    27: 'p10',
    28: 'p11',
    29: 'p12',
    30: 'p13',
    31: 'p14',
    32: 'p19',
    33: 'p20',
    34: 'p21',
    35: 'p23',
    36: 'p27',
    37: 'p28',
    38: 'p29',
    39: 'p30',
    40: 'p31',
    41: 'p32',
    42: 'p5',
    43: 'p6',
    44: 'p9',
    45: 'pa',
    46: 'pb',
    47: 'pdc',
    48: 'pg',
    49: 'pl1',
    50: 'pl10',
    51: 'pl100',
    52: 'pl110',
    53: 'pl120',
    54: 'pl130',
    55: 'pl15',
    56: 'pl2',
    57: 'pl20',
    58: 'pl25',
    59: 'pl3',
    60: 'pl30',
    61: 'pl35',
    62: 'pl4',
    63: 'pl40',
    64: 'pl5',
    65: 'pl50',
    66: 'pl55',
    67: 'pl60',
    68: 'pl7',
    69: 'pl70',
    70: 'pl8',
    71: 'pl80',
    72: 'pl90',
    73: 'pn',
    74: 'pne',
    75: 'pnl',
    76: 'pr0',
    77: 'pr100',
    78: 'pr20',
    79: 'pr30',
    80: 'pr40',
    81: 'pr50',
    82: 'pr60',
    83: 'pr70',
    84: 'pr80',
    85: 'ps',
    86: 'w10',
    87: 'w13',
    88: 'w15',
    89: 'w16',
    90: 'w20',
    91: 'w21',
    92: 'w22',
    93: 'w24',
    94: 'w3',
    95: 'w30',
    96: 'w31',
    97: 'w32',
    98: 'w34',
    99: 'w39',
    100: 'w40',
    101: 'w41',
    102: 'w42',
    103: 'w43',
    104: 'w45',
    105: 'w46',
    106: 'w47',
    107: 'w55',
    108: 'w56',
    109: 'w57',
    110: 'w63',
    111: 'w66',
    112: 'w68',
    113: 'w69',
    114: 'zo'}


class YOLOLayerNp():

    def __init__(self, anchors, num_classes=1, img_dim=1216):
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.num_samples = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.exp = np.exp  # grid size
        self.stride = None
        self.grid_x = None
        self.grid_y = None
        self.scaled_anchors = None
        self.anchor_w = None
        self.anchor_h = None

    def __call__(self, x):

        num_samples = x.shape[0]
        self.num_samples = num_samples
        grid_size = x.shape[2]
        prediction_ = (
            x.reshape(
                num_samples,
                self.num_anchors,
                self.num_classes + 5,
                grid_size,
                grid_size) .transpose(
                0,
                1,
                3,
                4,
                2))
        x = self.sigmoid_(prediction_[..., 0])  # Center x
        y = self.sigmoid_(prediction_[..., 1])  # Center y
        w_ = prediction_[..., 2]  # Width
        h_ = prediction_[..., 3]  # Height
        pred_conf = self.sigmoid_(prediction_[..., 4])  # Conf
        pred_cls = self.sigmoid_(prediction_[..., 5:])  # Cls pred.

        self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        pred_boxes = np.zeros(prediction_[..., :4].shape)
        pred_boxes[..., 0] = x + self.grid_x
        pred_boxes[..., 1] = y + self.grid_y
        pred_boxes[..., 2] = np.exp(w_) * self.anchor_w
        pred_boxes[..., 3] = np.exp(h_) * self.anchor_h

        output = np.concatenate(
            (
                pred_boxes.reshape(num_samples, -1, 4) * self.stride,
                pred_conf.reshape(num_samples, -1, 1),
                pred_cls.reshape(num_samples, -1, self.num_classes),
            ),
            -1,
        )
        return output

    def sigmoid_(self, inpu_x):
        s = 1 / (1 + self.exp(-inpu_x))
        return s

    def compute_grid_offsets(self, grid_size_):
        self.grid_size = grid_size_
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size
        self.grid_x = np.arange(g).repeat(g, 0).reshape(
            [1, 1, g, g]).transpose(0, 1, 3, 2)
        self.grid_y = np.arange(g).repeat(
            g, 0).transpose().reshape([1, 1, g, g])
        self.scaled_anchors = np.array(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])

        self.anchor_w = self.scaled_anchors[:, 0:1].reshape(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].reshape(
            (1, self.num_anchors, 1, 1))


def pre_process(imagepath):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1216, 1216), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    image_for_infer = img.astype(np.float32)
    return image_for_infer


def rescale_boxes(boxes, current_dim, original_shape):

    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def bbox_iou(box1, box2, x1y1x2y2=True):

    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1,
                         0,
                         10000000) * np.clip(inter_rect_y2 - inter_rect_y1 + 1,
                                             0,
                                             10000000)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def xywh_xyxy(x):
    y = np.zeros(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction_tmp, conf_thres=0.8, nms_thres=0.4):

    prediction_tmp[..., :4] = xywh_xyxy(prediction_tmp[..., :4])
    output = [None for _ in range(len(prediction_tmp))]
    for image_i, image_pred in enumerate(prediction_tmp):

        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        if not image_pred.shape[0]:
            continue
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        image_pred = image_pred[(-score).argsort()]
        class_confs = np.expand_dims(np.max(image_pred[:, 5:], axis=1), axis=1)
        class_preds = np.expand_dims(
            np.argmax(image_pred[:, 5:], axis=1), axis=1)
        detections_tmp = np.concatenate(
            (image_pred[:, :5], class_confs, class_preds), 1)
        keep_boxes = []
        while detections_tmp.shape[0]:
            large_overlap = bbox_iou(np.expand_dims(
                detections_tmp[0, :4], axis=0), detections_tmp[:, :4]) > nms_thres
            label_match = detections_tmp[0, -1] == detections_tmp[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and
            # matching labels
            invalid = large_overlap & label_match
            weights = detections_tmp[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections_tmp[0, :4] = (
                weights * detections_tmp[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections_tmp[0]]
            detections_tmp = detections_tmp[~invalid]
        if keep_boxes:
            output[image_i] = np.stack(keep_boxes)

    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default='./data/detection/data/test',
        help="path to dataset")
    parser.add_argument(
        "--save_json",
        type=str,
        default="./data/detection/Tinghua100K_result_for_test.json",
        help="path to dataset")
    parser.add_argument(
        "--save_image",
        type=str,
        default='./data/detection/result',
        help="path to dataset")
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.8,
        help="object confidence threshold")
    parser.add_argument("--draw", type=bool, default=True,
                        help="object confidence threshold")
    parser.add_argument(
        "--nms_thres",
        type=float,
        default=0.4,
        help="iou thresshold for non-maximum suppression")
    parser.add_argument(
        "--detection_pipline",
        type=str,
        default="./pipeline/detection.pipeline",
        help="object confidence threshold")
    parser.add_argument(
        "--class_pipline",
        type=str,
        default="./pipeline/class.pipeline",
        help="iou thresshold for non-maximum suppression")
    parser.add_argument(
        "--img_size",
        type=int,
        default=1216,
        help="size of each image dimension")
    opt = parser.parse_args()

    try:
        shutil.rmtree(opt.save_image)
        os.remove(opt.save_json)
    except BaseException:
        pass
    os.mkdir(opt.save_image)
    Image_ = [os.path.join(opt.image_folder, files)
              for files in os.listdir(opt.image_folder)]
    if len(Image_) == 0:
        print("the image is null")
        exit()
    streamManagerApi = StreamManagerApi()
    YOLOLayer1_np = YOLOLayerNp(anchors=[(54, 53), (72, 76), (118, 130)])
    YOLOLayer2_np = YOLOLayerNp(anchors=[(28, 28), (34, 34), (42, 43)])
    YOLOLayer3_np = YOLOLayerNp(anchors=[(10, 12), (17, 18), (22, 23)])
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # 构建pipeline
    with open(opt.detection_pipline, 'rb') as f:
        pipelineStr = f.read()

    with open(opt.class_pipline, 'rb') as f:
        pipelineStr_class = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr_class)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    dataInput = MxDataInput()

    train_results = {"imgs": {}}
    for k, img_path in enumerate(Image_[:]):
        img_id = img_path.split('/')[-1].split('.')[0]
        if os.path.exists(img_path) != 1:
            print("The test image does not exist.")

        STREAMNAME = b'detection'
        INPLUGINLD = 0

        post_img = pre_process(img_path)
        vision_tmp_list = MxpiDataType.MxpiVisionList()
        visio_Vec_tmp = vision_tmp_list.visionVec.add()
        vision_Info_tmp = visio_Vec_tmp.visionInfo
        vision_Info_tmp.width = 1216
        vision_Info_tmp.height = 1216
        vision_Info_tmp.widthAligned = 1216
        vision_Info_tmp.heightAligned = 1216
        vision_data_tmp = visio_Vec_tmp.visionData
        vision_data_tmp.dataStr = post_img.tobytes()
        vision_data_tmp.deviceId = 0
        vision_data_tmp.memType = 0
        vision_data_tmp.dataSize = len(post_img)
        vision_data_tmp.dataSize = 1

        protobuf_Vec_tmp = InProtobufVector()
        proto_buf_tmp = MxProtobufIn()
        KEY_VALUE = b"appsrc0"
        proto_buf_tmp.key = KEY_VALUE
        proto_buf_tmp.type = b"MxTools.MxpiVisionList"
        proto_buf_tmp.protobuf = vision_tmp_list.SerializeToString()
        protobuf_Vec_tmp.push_back(proto_buf_tmp)
        unique_Id_tmp = streamManagerApi.SendProtobuf(STREAMNAME, INPLUGINLD,
                                                      protobuf_Vec_tmp)

        if unique_Id_tmp < 0:
            print("Failed to send data to stream.")
            exit()
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_result = streamManagerApi.GetProtobuf(STREAMNAME, 0, keyVec)
        if infer_result.size() == 0:
            print("infer_result is null")
            continue

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            continue

        tenList_rmp = MxpiDataType.MxpiTensorPackageList()
        tenList_rmp.ParseFromString(infer_result[0].messageBuf)
        yolo_outputs_np = []
        prediction0 = np.frombuffer(
            tenList_rmp.tensorPackageVec[0].tensorVec[0].dataStr,
            dtype=np.float32)
        prediction_shape0 = tenList_rmp.tensorPackageVec[0].tensorVec[0].tensorShape
        prediction0 = np.reshape(prediction0, prediction_shape0)
        yolo_outputs_np.append(YOLOLayer1_np(prediction0))

        prediction1 = np.frombuffer(
            tenList_rmp.tensorPackageVec[0].tensorVec[1].dataStr,
            dtype=np.float32)
        prediction_shape1 = tenList_rmp.tensorPackageVec[0].tensorVec[1].tensorShape
        prediction1 = np.reshape(prediction1, prediction_shape1)
        yolo_outputs_np.append(YOLOLayer2_np(prediction1))

        prediction2 = np.frombuffer(
            tenList_rmp.tensorPackageVec[0].tensorVec[2].dataStr,
            dtype=np.float32)
        prediction_shape2 = tenList_rmp.tensorPackageVec[0].tensorVec[2].tensorShape
        prediction2 = np.reshape(prediction2, prediction_shape2)
        yolo_outputs_np.append(YOLOLayer3_np(prediction2))

        detections = np.concatenate(yolo_outputs_np, 1)
        detections = non_max_suppression(
            detections, opt.conf_thres, opt.nms_thres)[0]

        if detections is not None:
            objects = []
            img_copy = Image.open(img_path)
            detections = rescale_boxes(detections, opt.img_size, img_copy.size)

            for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(
                    detections):

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w >= 10 and box_h >= 10:
                    crop_sign_org = img_copy.crop(
                        (x1, y1, x2, y2)).convert(
                        mode="RGB")

                    if min(crop_sign_org.size) < 32:
                        w, h = crop_sign_org.size
                        n = 32 // min(crop_sign_org.size) + 1
                        crop_sign_org = crop_sign_org.resize((w * n, h * n))

                    time.sleep(0.05)
                    if opt.draw:
                        draw = ImageDraw.Draw(img_copy)  # 在上面画画
                        draw.rectangle([x1, y1, x2, y2],
                                       outline=(255, 0, 0), width=5)

            if opt.draw:
                img_path_save = os.path.join(
                    opt.save_image, str(img_id) + '.png')
                img_copy.save(img_path_save)
        else:
            print('detection is null')