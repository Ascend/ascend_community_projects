import numpy as np
import cv2


class BBoxUtility(object):
    def __init__(self):
        pass
    
    def sigmoid(self, z):
        """sigmoid 激活函数"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def decode_boxes(self, pred_box, anchors, variances = [0.1, 0.2]):
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
        A = box_a.shape[1]
        B = box_b.shape[1]

        max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, :, 2:], axis=2), (n, A, B, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, :, 2:], axis=1), (n, A, B, 2)))

        min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :, :2], axis=2), (n, A, B, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, :, :2], axis=1), (n, A, B, 2)))
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
        inter = inter[:, :, :, 0] * inter[:, :, :, 1]

        area_a = np.broadcast_to(np.expand_dims(((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])), axis=2), inter.shape)  # [A,B]
        area_b = np.broadcast_to(np.expand_dims(((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])), axis=1), inter.shape)  # [A,B]
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)

    def fast_non_max_suppression(self, box_thre, class_thre, mask_thre, nms_iou=0.5, top_k=200, max_detections=100):
        #---------------------------------------------------------#
        #   先进行tranpose，方便后面的处理
        #   [80, num_of_kept_boxes]
        #---------------------------------------------------------#
        class_thre      = class_thre.transpose(1, 0)
        class_thre      = np.ascontiguousarray(class_thre)
        #---------------------------------------------------------#
        #   [80, num_of_kept_boxes]
        #   每一行坐标为该种类所有的框的得分，
        #   对每一个种类单独进行排序
        #---------------------------------------------------------#
        idx = np.argsort(class_thre)
        idx = idx[:,::-1]
        class_thre = np.sort(class_thre, axis=1) 
        class_thre = class_thre[:,::-1]
        
        idx             = idx[:, :top_k]
        class_thre      = class_thre[:, :top_k]
        num_classes, num_dets = idx.shape
        #---------------------------------------------------------#
        #   将num_classes作为第一维度，对每一个类进行非极大抑制
        #   [80, num_of_kept_boxes, 4]    
        #   [80, num_of_kept_boxes, 32]    
        #---------------------------------------------------------#
        box_thre    = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)
        mask_thre   = mask_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)

        iou         = self.jaccard(box_thre, box_thre)
        #---------------------------------------------------------#
        #   [80, num_of_kept_boxes, num_of_kept_boxes]
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
        return box_nms, class_nms, class_ids, mask_nms

    def yolact_correct_boxes(self, boxes, image_shape):
        image_size          = np.array(image_shape)[::-1]

        scales              = np.concatenate([image_size, image_size], axis=-1)
        boxes               = boxes * scales
        boxes[:, [0, 1]]    = np.minimum(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [2, 3]]    = np.maximum(boxes[:, [0, 1]], boxes[:, [2, 3]])
        boxes[:, [0, 1]]    = np.maximum(boxes[:, [0, 1]], np.zeros_like(boxes[:, [0, 1]]))
        boxes[:, [2, 3]]    = np.minimum(boxes[:, [2, 3]], np.broadcast_to(np.expand_dims(image_size, axis=0), (boxes.shape[0], 2)))
        return boxes

    def crop(self, masks, boxes):
        h, w, n     = masks.shape
        x1, x2      = boxes[:, 0], boxes[:, 2]
        y1, y2      = boxes[:, 1], boxes[:, 3]

        rows        = np.broadcast_to(np.arange(w, dtype=x1.dtype).reshape(1, -1, 1), (h, w, n))
        cols        = np.broadcast_to(np.arange(h, dtype=x1.dtype).reshape(-1, 1, 1), (h, w, n))

        masks_left  = rows >= x1.reshape(1, 1, -1)
        masks_right = rows < x2.reshape(1, 1, -1)
        masks_up    = cols >= y1.reshape(1, 1, -1)
        masks_down  = cols < y2.reshape(1, 1, -1)

        crop_mask   = masks_left * masks_right * masks_up * masks_down
        return masks * crop_mask.astype(np.float32)

    def decode_nms(self, outputs, anchors, confidence, nms_iou, image_shape, traditional_nms=False, max_detections=100):
        #---------------------------------------------------------#
        #   pred_box    [18525, 4]  对应每个先验框的调整情况
        #   pred_class  [18525, 81] 对应每个先验框的种类      
        #   pred_mask   [18525, 32] 对应每个先验框的语义分割情况
        #   pred_proto  [128, 128, 32]  需要和结合pred_mask使用
        #---------------------------------------------------------#
        pred_box    = outputs[0].squeeze()
        pred_class  = outputs[1].squeeze()
        pred_masks  = outputs[2].squeeze()
        pred_proto  = outputs[3].squeeze()

        #---------------------------------------------------------#
        #   将先验框调整获得预测框，
        #   [18525, 4] boxes是左上角、右下角的形式。
        #---------------------------------------------------------#
        boxes       = self.decode_boxes(pred_box, anchors)
        #---------------------------------------------------------#
        #   除去背景的部分，并获得最大的得分 
        #   [18525, 80]
        #   [18525]
        #---------------------------------------------------------#
        # print('!!!!!!!!!!!!!!!!!!!!!!!')
        # print(pred_class.shape)
        # print('@@@@@@@@@@@@@@@@@@@@@@')
        pred_class          = pred_class[:, 1:]    
        pred_class_max = np.max(pred_class, 1)
        # print('######################')
        # print(pred_class_max)
        # print('$$$$$$$$$$$$$$$$$$$$')
        keep        = (pred_class_max > confidence)
        #---------------------------------------------------------#
        #   保留满足得分的框，如果没有框保留，则返回None
        #---------------------------------------------------------#
        box_thre    = boxes[keep, :]
        class_thre  = pred_class[keep, :]
        mask_thre   = pred_masks[keep, :]
        if class_thre.shape[0] == 0:
            return None, None, None, None, None
        if not traditional_nms:
            box_thre, class_thre, class_ids, mask_thre = self.fast_non_max_suppression(box_thre, class_thre, mask_thre, nms_iou)
            keep        = class_thre > confidence
            box_thre    = box_thre[keep]
            class_thre  = class_thre[keep]
            class_ids   = class_ids[keep]
            mask_thre   = mask_thre[keep]
            
        box_thre    = self.yolact_correct_boxes(box_thre, image_shape)
        #---------------------------------------------------------#
        #   pred_proto      [128, 128, 32]
        #   mask_thre       [num_of_kept_boxes, 32]
        #   masks_sigmoid   [128, 128, num_of_kept_boxes]
        #---------------------------------------------------------#
        masks_sigmoid   = self.sigmoid(np.matmul(pred_proto, np.transpose(mask_thre)))
        #----------------------------------------------------------------------#
        #   masks_sigmoid   [image_shape[0], image_shape[1], num_of_kept_boxes]
        #----------------------------------------------------------------------#
        masks_sigmoid   = cv2.resize(masks_sigmoid, (image_shape[1], image_shape[0]),  interpolation=cv2.INTER_LINEAR)
        if masks_sigmoid.ndim == 2:
            masks_sigmoid   = np.expand_dims(masks_sigmoid, axis=2)
        masks_sigmoid   = np.ascontiguousarray(masks_sigmoid)
        masks_sigmoid   = self.crop(masks_sigmoid, box_thre)
        #----------------------------------------------------------------------#
        #   masks_arg   [image_shape[0], image_shape[1]]
        #   获得每个像素点所属的实例
        #----------------------------------------------------------------------#
        masks_arg       = np.argmax(masks_sigmoid, axis=-1)
        #----------------------------------------------------------------------#
        #   masks_arg   [image_shape[0], image_shape[1], num_of_kept_boxes]
        #   判断每个像素点是否满足门限需求
        #----------------------------------------------------------------------#
        masks_sigmoid   = masks_sigmoid > 0.5

        return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid

