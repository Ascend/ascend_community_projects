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

import numpy as np
import cv2


class BBoxUtility(object):
    def __init__(self):
        pass
    
    def sigmoid(self, z):
        """sigmoid 激活函数"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def decode_boxes(self, pred_box, anchors, variances):
        #---------------------------------------------------------#
        #   anchors[:, :2] 先验框中心
        #   anchors[:, 2:] 先验框宽高
        #   对先验框的中心和宽高进行调整，获得预测框
        #---------------------------------------------------------#
        # boxes = torch.cat((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:],
        #                 anchors[:, 2:] * torch.exp(pred_box[:, 2:] * variances[1])), 1)
        exp = np.exp(pred_box[:, 2:] * variances[1])
        boxes = np.concatenate((anchors[:, :2] + pred_box[:, :2] * variances[0] * anchors[:, 2:],
                        anchors[:, 2:] * exp), axis = 1)
        #---------------------------------------------------------#
        #   获得左上角和右下角
        #---------------------------------------------------------#
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        use_batch = True
        if box_a.ndim == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]

        n = box_a.shape[0]
        a = box_a.shape[1]
        b = box_b.shape[1]

        max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, :, 2:], axis=2), (n, a, b, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, :, 2:], axis=1), (n, a, b, 2)))

        min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :, :2], axis=2), (n, a, b, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, :, :2], axis=1), (n, a, b, 2)))
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
        inter = inter[:, :, :, 0] * inter[:, :, :, 1]

        area_a = np.broadcast_to(
            np.expand_dims(((box_a[:, :, 2] - box_a[:, :, 0])*(box_a[:, :, 3] - box_a[:, :, 1])), axis=2), inter.shape)
        area_b = np.broadcast_to(
            np.expand_dims(((box_b[:, :, 2] - box_b[:, :, 0])*(box_b[:, :, 3] - box_b[:, :, 1])), axis=1), inter.shape)
        union = area_a + area_b - inter
        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)

    def fast_non_max_suppression(self, box_thre, class_thre, mask_thre, nms_iou=0.5, top_k=200, max_detections=100):
        #---------------------------------------------------------#
        #   先进行tranpose，方便后面的处理
        #---------------------------------------------------------#
        class_thre      = class_thre.transpose(1, 0)
        class_thre      = np.ascontiguousarray(class_thre)
        #---------------------------------------------------------#
        #   每一行坐标为该种类所有的框的得分，
        #   对每一个种类单独进行排序
        #---------------------------------------------------------#
        idx = np.argsort(class_thre)
        idx = idx[:, ::-1]
        class_thre = np.sort(class_thre, axis=1) 
        class_thre = class_thre[:, ::-1]
        
        idx             = idx[:, :top_k]
        class_thre      = class_thre[:, :top_k]
        num_classes, num_dets = idx.shape
        #---------------------------------------------------------#
        #   将num_classes作为第一维度，对每一个类进行非极大抑制 
        #---------------------------------------------------------#
        box_thre    = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)
        mask_thre   = mask_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)

        iou         = self.jaccard(box_thre, box_thre)
        #---------------------------------------------------------#
        #   取矩阵的上三角部分
        #---------------------------------------------------------#
        iou_max = np.amax(np.triu(iou, 1), axis=1)

        #---------------------------------------------------------#
        #   获取和高得分重合程度比较低的预测结果
        #---------------------------------------------------------#
        keep        = (iou_max <= nms_iou)
        class_ids   = np.broadcast_to(np.arange(num_classes)[:, None], keep.shape)

        box_nms     = box_thre[keep]
        class_nms   = class_thre[keep]
        class_ids   = class_ids[keep]
        mask_nms    = mask_thre[keep]
        
        idx = np.argsort(class_nms, axis=0)
        idx = idx[::-1]
        idx         = idx[:max_detections]
        box_nms     = box_nms[idx]
        class_nms   = class_nms[idx]
        class_ids   = class_ids[idx]
        mask_nms    = mask_nms[idx]
        return [box_nms, class_nms, class_ids, mask_nms]

    def correct_boxes(self, boxes, shape):
        size          = np.array(shape)[::-1]

        scales              = np.concatenate([size, size], axis=-1)
        boxes               = boxes * scales
        boxes[:, [0, 1]]    = np.minimum(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [2, 3]]    = np.maximum(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [0, 1]]    = np.maximum(boxes[:, [0, 1]], np.zeros_like(boxes[:, [0, 1]]))
        boxes[:, [2, 3]]    = np.minimum(boxes[:, [2, 3]], np.broadcast_to(
                                    np.expand_dims(size, axis=0), (boxes.shape[0], 2)))
        return boxes

    def crop(self, ms, boxes):
        h, w, n     = ms.shape
        x1, x2      = boxes[:, 0], boxes[:, 2]
        y1, y2      = boxes[:, 1], boxes[:, 3]

        ro        = np.broadcast_to(np.arange(w, dtype=x1.dtype).reshape(1, -1, 1), (h, w, n))
        co        = np.broadcast_to(np.arange(h, dtype=x1.dtype).reshape(-1, 1, 1), (h, w, n))

        left  = ro >= x1.reshape(1, 1, -1)
        right = ro < x2.reshape(1, 1, -1)
        up    = co >= y1.reshape(1, 1, -1)
        down  = co < y2.reshape(1, 1, -1)

        mask   = left * right * up * down
        return ms * mask.astype(np.float32)

    def decode_nms(self, puts, an, confidence, nms_iou, image_shape, traditional_nms=False, max_detections=100):
        
        box    = puts[0].squeeze()
        p_class  = puts[1].squeeze()
        p_masks  = puts[2].squeeze()
        p_proto  = puts[3].squeeze()

        #---------------------------------------------------------#
        #   将先验框调整获得预测框，
        #   [18525, 4] boxes是左上角、右下角的形式。
        #---------------------------------------------------------#
        boxesss       = self.decode_boxes(box, an, [0.1, 0.2])
        #---------------------------------------------------------#
        #   除去背景的部分，并获得最大的得分 
        #---------------------------------------------------------#
        p_class          = p_class[:, 1:]    
        pred_cl_max = np.max(p_class, 1)
        keep        = (pred_cl_max > confidence)
        #---------------------------------------------------------#
        #   保留满足得分的框，如果没有框保留，则返回None
        #---------------------------------------------------------#
        box_th    = boxesss[keep, :]
        class_th  = p_class[keep, :]
        m_thre   = p_masks[keep, :]
        if class_th.shape[0] == 0:
            return [None, None, None, None, None]
        if not traditional_nms:
            box_th, class_th, cla_ids, m_thre = self.fast_non_max_suppression(box_th,
                                                                                    class_th, m_thre, nms_iou)
            keep        = class_th > confidence
            box_th    = box_th[keep]
            class_th  = class_th[keep]
            cla_ids   = cla_ids[keep]
            m_thre   = m_thre[keep]
            
        b_thre    = self.correct_boxes(box_th, image_shape)
        masks_sig   = self.sigmoid(np.matmul(p_proto, np.transpose(m_thre)))
        masks_sig   = cv2.resize(masks_sig, (image_shape[1], image_shape[0]),  interpolation=cv2.INTER_LINEAR)
        if masks_sig.ndim == 2:
            masks_sig   = np.expand_dims(masks_sig, axis=2)
        masks_sig   = np.ascontiguousarray(masks_sig)
        masks_sig   = self.crop(masks_sig, b_thre)
        #----------------------------------------------------------------------#
        #   获得每个像素点所属的实例
        #----------------------------------------------------------------------#
        m_arg       = np.argmax(masks_sig, axis=-1)
        #----------------------------------------------------------------------#
        #   判断每个像素点是否满足门限需求
        #----------------------------------------------------------------------#
        masks_sig   = masks_sig > 0.5

        return [b_thre, class_th, cla_ids, m_arg, masks_sig]

