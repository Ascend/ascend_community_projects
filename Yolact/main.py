#!/usr/bin/env python
# coding=utf-8

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

from __future__ import annotations
import argparse
import os
import os.path as osp
import colorsys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.utils_bbox import BBoxUtility
from utils.anchors import get_anchors
from utils.utils_map import MakeJson, prep_metrics
from utils.utils import cvtcolor, resize_image, get_classes, get_coco_label_map, preprocess_input
import cv2
from data import cfg
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
from StreamManagerApi import StreamManagerApi, StringVector, InProtobufVector, MxProtobufIn


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='YOLACT Inferring sdk')
    # Datasets
    parser.add_argument('--image', default=None, type=str,
                        help='An input folder of images and output folder to save detected images.'
                                'Should be in the format input->output.')
    parser.add_argument('--PL_PATH', default='./model_data/yolact.pipeline', type=str,
                        help='pipeline path')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])') 
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')    
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')    
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--score_threshold', default=0.3, type=float,
                        help='Detections with a score under this threshold will not be considered.'
                                'This currently only works in display mode.')
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--confidence', default=0.05, type=float,
                        help='confidence of threshold')
    parser.add_argument('--nms_iou', default=0.5, type=float,
                        help='nms_iou of threshold')
    parser.add_argument('--traditional_nms', default=False, type=str2bool,
                        help='traditional_nms')
    parser.add_argument('--classes_path', default='./model_data/coco_classes.txt', type=str,
                        help='classes path')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False,
                        output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False,
                        crop=True, detect=False, display_fps=False,
                        emulate_playback=False)
    # pca config
    par_args = parser.parse_args()

    return par_args

iou_thresholds = [x / 100 for x in range(50, 100, 5)]


def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    for i in range(tensor.shape[0]):
        da = np.expand_dims(tensor[i, :], 0)
        tensor_pack = tensor_package_list.tensorPackageVec.add()
        ten_vec = tensor_pack.tensorVec.add()
        ten_vec.deviceId = 0
        ten_vec.memType = 0
        ten_vec.tensorShape.extend(da.shape)
        ten_vec.dataStr = da.tobytes()
        ten_vec.tensorDataSize = da.shape[0]
    keys = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobu_vec = InProtobufVector()
    protobu = MxProtobufIn()
    protobu.key = keys
    protobu.type = b'MxTools.MxpiTensorPackageList'
    protobu.protobuf = tensor_package_list.SerializeToString()
    protobu_vec.push_back(protobu)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobu_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    print('succes')
    return True


def evalimage(stream_manager_api, path:str, save_path:str=None):
    image = Image.open(path)
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtcolor(image)
    image_origin = np.array(image, np.uint8)
    image_data      = resize_image(image, (544, 544))
    batch      = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    batch = batch.astype(np.float32)
    stream_name = b'im_yolact'
    in_plugin_id = 0
    if not send_source_data(0, batch, stream_name, stream_manager_api):
        return None
    keys = [b"mxpi_tensorinfer0"]
    key_vec = StringVector()
    for key in keys:
        key_vec.push_back(key)
    infer_results = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, key_vec)
    if infer_results.size() == 0 or infer_results.size() == 0:
        print("infer_result is null")
        exit()
    if infer_results[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_results[0].errorCode, infer_results[0].data.decode()))
        exit()
    result_list = MxpiDataType.MxpiTensorPackageList()
    result_list.ParseFromString(infer_results[0].messageBuf)
    pred_boxes = np.array(
        [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)])
    pred_boxes = pred_boxes.reshape(1, 18525, 4)
    pred_classes = np.array(
        [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)])
    pred_classes = pred_classes.reshape(1, 18525, 81)
    pred_masks = np.array(
        [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32)])
    pred_masks = pred_masks.reshape(1, 18525, 32)
    pred_proto = np.array(
        [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[3].dataStr, dtype=np.float32)])
    pred_proto = pred_proto
    detect = BBoxUtility().reshape(1, 136, 136, 32)
    outputs = tuple([pred_boxes, pred_classes, pred_masks, pred_proto])
            
    #----------------------------------------------------------------------#
    #   根据每个像素点所属的实例和是否满足门限需求，判断每个像素点的种类
    #----------------------------------------------------------------------#
    class_names, num_classes  = get_classes(args.classes_path)
    num_classes                    += 1
    anchors                        = get_anchors([544, 544], [24, 48, 96, 192, 384])
    
    #---------------------------------------------------#
    #   画框设置不同的颜色
    #---------------------------------------------------#
    if num_classes <= 81:
        colors = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], 
                            [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], 
                            [20, 55, 200], [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], 
                            [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], 
                            [90, 155, 50], [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], 
                            [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], 
                            [90, 125, 120], [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], 
                            [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], 
                            [198, 75, 20], [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], 
                            [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], 
                            [18, 185, 90], [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], 
                            [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], 
                            [18, 25, 190], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
                            [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], 
                            [155, 0, 255], [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], 
                            [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]],
                            dtype='uint8')
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
    confidence = 0.5
    nms_iou = 0.3
    results = detect.decode_nms(outputs, anchors, confidence, nms_iou, image_shape, args.traditional_nms)
    if results[0] is None:
        return image
    box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = [x for x in results]
    
    masks_class     = masks_sigmoid * (class_ids[None, None, :] + 1)
    masks_class     = np.reshape(masks_class, [-1, np.shape(masks_sigmoid)[-1]])
    masks_class     = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]),
                                np.reshape(masks_arg, [-1])], [image_shape[0], image_shape[1]])
    #---------------------------------------------------------#
    #   设置字体与边框厚度
    #---------------------------------------------------------#
    scale       = 0.6
    thickness   = int(max((image.size[0] + image.size[1]) // np.mean([544, 544]), 1))
    font        = cv2.FONT_HERSHEY_DUPLEX
    color_masks     = colors[masks_class].astype('uint8')
    image_fused     = cv2.addWeighted(color_masks, 0.4, image_origin, 0.6, gamma=0)
    for i in range(np.shape(class_ids)[0]):
        left, top, right, bottom = np.array(box_thre[i, :], np.int32)
        #---------------------------------------------------------#
        #   获取颜色并绘制预测框
        #---------------------------------------------------------#
        color = colors[class_ids[i] + 1].tolist()
        cv2.rectangle(image_fused, (left, top), (right, bottom), color, thickness)
        #---------------------------------------------------------#
        #   获得这个框的种类并写在图片上
        #---------------------------------------------------------#
        class_name  = class_names[class_ids[i]]
        text_str    = f'{class_name}: {class_thre[i]:.2f}'
        text_w, text_h = cv2.getTextSize(text_str, font, scale, 1)[0]
        cv2.rectangle(image_fused, (left, top), (left + text_w, top + text_h + 5), color, -1)
        cv2.putText(image_fused, text_str, (left, top + 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    image = Image.fromarray(np.uint8(image_fused))
    image.save("img.jpg")
    return None


def val(val_args):
    # init streams
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(val_args.PL_PATH, 'rb') as pl:
        pipeline_str = pl.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    annotationfile = './data/coco/annotations/instances_val2017.json'
    coco_gt = COCO(annotationfile)
    imagefolder = './data/coco/images'
    image_ids = list(coco_gt.imgToAnns.keys())
    #-------------------------------------------------------#
    #   获得测试用的图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #-------------------------------------------------------#
    json_path       = "./data/coco/annotations/instances_val2017.json"
    map_out_path    = 'map_out'
    test       = COCO(json_path)
    class_names, _  = get_classes(val_args.classes_path)
    coco_label_map  = get_coco_label_map(test, class_names)
    
    if val_args.image is not None:
        if ':' in val_args.image:
            inp, out = val_args.image.split(':')
            evalimage(stream_manager_api, inp, out)
        else:
            evalimage(stream_manager_api, val_args.image)
        return
   
    if not osp.exists(map_out_path):
        os.makedirs(map_out_path)
    print("Get predict result.")
    m_json   = MakeJson(map_out_path, coco_label_map)
    image_ids = image_ids[4950:]
    for image_idx, image_id in enumerate(image_ids):
        print('image_idx = %d image_id = %d.' % (image_idx, image_id))
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = os.path.join(imagefolder, image_info['file_name'])
           
        image       = Image.open(image_path)
        image_shape = np.array(np.shape(image)[0:2])
        image           = cvtcolor(image)
        image_data      = resize_image(image, (544, 544))
        image_data      = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
              
        print('Detect image: ', image_idx, ': ', image_info['file_name'],
              ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()
        batch = image_data 
        batch = batch.astype(np.float32)
        stream_name = b'im_yolact'
        in_plugin_id = 0

        if not send_source_data(0, batch, stream_name, stream_manager_api):
            return
        keys = [b"mxpi_tensorinfer0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)
        infer_results = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, key_vec)
        if infer_results.size() == 0 or infer_results.size() == 0:
            print("infer_result is null")
            exit()
        if infer_results[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_results[0].errorCode, infer_results[0].data.decode()))
            exit()
        result_list = MxpiDataType.MxpiTensorPackageList()
        result_list.ParseFromString(infer_results[0].messageBuf)
        pred_boxes = np.array(
            [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)])
        pred_boxes = pred_boxes.reshape(1, 18525, 4)
        pred_classes = np.array(
            [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)])
        pred_classes = pred_classes.reshape(1, 18525, 81)
        pred_masks = np.array(
            [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32)])
        pred_masks = pred_masks.reshape(1, 18525, 32)
        pred_proto = np.array(
            [np.frombuffer(result_list.tensorPackageVec[0].tensorVec[3].dataStr, dtype=np.float32)])
        pred_proto = pred_proto.reshape(1, 136, 136, 32)
        anchor = get_anchors([544, 544], [24, 48, 96, 192, 384])
        detect = BBoxUtility()
        outputs = tuple([pred_boxes, pred_classes, pred_masks, pred_proto])
        results = detect.decode_nms(outputs, anchor, val_args.confidence, val_args.nms_iou,
                                        image_shape, val_args.traditional_nms)
        if results[0] is None:
            continue
        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = [x for x in results]
        if box_thre is None:
            continue
        prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, image_id, m_json)

    m_json.dump()
    print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')

    bbox = test.loadRes(osp.join(map_out_path, "bbox_detections.json"))
    mask = test.loadRes(osp.join(map_out_path, "mask_detections.json"))
    print('\nBBoxes:')
    b_eval = COCOeval(test, bbox, 'bbox')
    b_eval.evaluate()
    b_eval.accumulate()
    b_eval.summarize()
    print('\nMasks:')
    b_eval = COCOeval(test, mask, 'segm')
    b_eval.evaluate()
    b_eval.accumulate()
    b_eval.summarize()

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')
    val(val_args=args)
