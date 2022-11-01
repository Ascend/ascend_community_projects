""" Contains functions used to sanitize and prepare the output of Yolact. """

import numpy as np
import cv2

from data import cfg, mask_type, MEANS, STD
from utils import timer
from .box_utils import crop, sanitize_coordinates

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    dets = det_output[batch_idx]
    
    if dets is None:
        return [numpy.empty()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]

        if dets['score'].size == 0:
            return [numpy.empty()] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    # Undo the padding introduced with preserve_aspect_ratio
    if cfg.preserve_aspect_ratio:
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)

        # Get rid of any detections whose centers are outside the image
        boxes = dets['box']
        boxes = center_size(boxes)
        s_w, s_h = (r_w/cfg.max_size, r_h/cfg.max_size)

        not_outside = ((boxes[:, 0] > s_w) + (boxes[:, 1] > s_h)) < 1 # not (a or b)
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][not_outside]

        # A hack to scale the bboxes to the right size
        b_w, b_h = (cfg.max_size / r_w * w, cfg.max_size / r_h * h)

    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']

        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())

        if visualize_lincomb:
            display_lincomb(proto_data, masks)

        masks = np.matmul(proto_data, masks.T)
        masks = cfg.mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.transpose(2, 0, 1)
        masks = np.ascontiguousarray(masks)

        # Scale masks up to the full image
        if cfg.preserve_aspect_ratio:
            # Undo padding
            masks = masks[:, :int(r_h/cfg.max_size*proto_data.size(1)), :int(r_w/cfg.max_size*proto_data.size(2))]

        masks1= []
        for i in range(masks.shape[0]):
            masks1.append(cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_LINEAR))
        masks = np.array(masks1)
        # Binarize the masks
        cv2.threshold(masks,0.5,1,cv2.THRESH_BINARY)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = boxes.astype(int)
    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue

            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask

        masks = full_masks

    return classes, scores, boxes, masks
