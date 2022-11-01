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
# ============================================================================
import argparse
import os
import os.path as osp
# from data import COCODetection, get_label_map, MEANS, COLORS
import colorsys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.augmentations import BaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils.utils_bbox import BBoxUtility                         #######method2 
from utils.anchors import get_anchors                            #############method2 
from utils.utils_map import Make_json, prep_metrics
from utils.utils import cvtColor, get_classes, get_coco_label_map
import cv2
from data import cfg, MEANS, STD, COLORS
import numpy as np
from utils import timer
from scipy import interpolate
import MxpiDataType_pb2 as MxpiDataType
from itertools import product
from math import sqrt
from utils.functions import SavePath
from layers.output_utils import postprocess
from layers.box_utils import mask_iou, jaccard
import pickle
from collections import OrderedDict
from PIL import Image
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput, InProtobufVector, MxProtobufIn
from layers import Detect


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
    parser.add_argument('--image', default='data/yolact_example_0.png:output_image.png', type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--PL_PATH', default='./yolact.pipeline', type=str,
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
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
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

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)
    # pca config
    args = parser.parse_args()

    return args

iou_thresholds = [x / 100 for x in range(50, 100, 5)]

class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape
        
        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        image = cv2.resize(image, (width, height))

        if self.resize_gt:
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))
            
            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width  / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, labels

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}
    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    print(all_maps) 
    print('bbbbb')
    print_maps(all_maps)
    exit()    
    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()

class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1

            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy)
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        #torch.cuda.synchronize()
    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][:args.top_k]
        classes, scores, boxes = [x[:args.top_k] for x in t[:3]]


    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break
    
    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])

        colors = np.concatenate([(np.array(get_color(j)) / 255 ).reshape(1, 1, 1, 3) for j in range(num_dets_to_consider)], axis=0)
        masks = np.tile(masks, [1,1,1,3])
        masks_color = masks * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(axis=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)

        img_gpu = img_gpu * inv_alph_masks.prod(axis=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255)
    
    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy

def get_label_map():
    if cfg.dataset.label_map is None:
        return {x+1: x+1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map 

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res

class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)

class COCODetection():
    def __init__(self, image_path, info_file, ids, transform=None,
                 target_transform=None,
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO
        
        if target_transform is None:
            target_transform = COCOAnnotationTransform()

        self.root = image_path
        self.coco = COCO(info_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.target_transform = COCOAnnotationTransform()
        
        self.name = dataset_name
        self.has_gt = has_gt
        self.ids=ids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []
        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd
        
        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path)
        height, width, _ = img.shape
        
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': target[:, 4]})
                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']
                
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None
        if target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))
        return img.transpose(2,0,1), target, masks, height, width, num_crowds

class FastBaseTransform():
    def __init__(self):
        self.mean = np.array(MEANS)[None, :, None, None]
        self.std = np.array(STD)[None, :, None, None]
        self.transform = cfg.backbone.transform
    def __call__(self, img):
        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.shape
            img_size = Resize.calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = (img_size[1], img_size[0]) # Pytorch needs h, w
        else:
            img_size = (cfg.max_size, cfg.max_size)
        img = img.transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        h, w = img_size
        img1=[]
        for i in range(img.shape[1]):
            img1.append(cv2.resize(img[0][i], (w, h), interpolation=cv2.INTER_LINEAR))
        img = np.array(img1)
        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        img = img[:, (2, 1, 0), :, :]
        img = np.ascontiguousarray(img)
        # Return value is in channel order [n, c, h, w] and RGB
        return img


def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret

def send_source_data(appsrc_id, tensor, stream_name, stream_manager):
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    for i in range(tensor.shape[0]):
        data = np.expand_dims(tensor[i,:],0)
        tensor_package = tensor_package_list.tensorPackageVec.add()
        tensor_vec = tensor_package.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0
        tensor_vec.tensorShape.extend(data.shape)
        tensor_vec.dataStr = data.tobytes()
        tensor_vec.tensorDataSize = data.shape[0]
    key = "appsrc{}".format(appsrc_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = stream_manager.SendProtobuf(stream_name, appsrc_id, protobuf_vec)
    if ret < 0:
        print("Failed to send data to stream.")
        return False
    print('succes')
    return True

def evalimage(stream_manager_api, path:str, save_path:str=None):
    image = Image.open(path)
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_origin = np.array(image, np.uint8)
    image_data      = resize_image(image, (544, 544))
    batch      = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    batch = batch.astype(np.float32)
    #exit()
    data_input = MxDataInput()
    stream_name=b'im_yolact'
    in_plugin_id=0
    if not send_source_data(0, batch, stream_name, stream_manager_api):
        return
    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)
    infer_results = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
    if infer_results.size() == 0 or infer_results.size() == 0:
        print("infer_result is null")
        exit()
    if infer_results[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_results[0].errorCode, infer_results[0].data.decode()))
        exit()
    resultList = MxpiDataType.MxpiTensorPackageList()
    resultList.ParseFromString(infer_results[0].messageBuf)
    pred_boxes = np.array(
        [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)]).reshape(1, 18525, 4)
    pred_classes = np.array(
        [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)]).reshape(1, 18525 ,81)
    pred_masks = np.array(
        [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32)]).reshape(1, 18525, 32)
    pred_proto = np.array(
        [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[3].dataStr, dtype=np.float32)]).reshape(1, 136, 136, 32)  
    # anchor = get_anchors([544, 544], [24, 48, 96, 192, 384])
    detect = BBoxUtility()
    outputs = tuple([pred_boxes, pred_classes, pred_masks, pred_proto])
    print('zzzz')
    # exit()
            
    #----------------------------------------------------------------------#
    #   masks_class [image_shape[0], image_shape[1]]
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
                            [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')
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
    masks_class     = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]), np.reshape(masks_arg, [-1])], [image_shape[0], image_shape[1]])
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

def resize_image(image, size):
    w, h    = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    mean    = (123.68, 116.78, 103.94)
    std     = (58.40, 57.12, 57.38)
    image   = (image - mean)/std
    return image
    ####################################
    
def val(args):
    cfg.mask_proto_debug = args.mask_proto_debug
    # init streams
    train_mode = False
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(args.PL_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    ANNOTATIONFILE = './data/coco/annotations/instances_val2017.json'
    coco_gt = COCO(ANNOTATIONFILE)
    IMAGEFOLDER = './data/coco/images'
    image_ids = list(coco_gt.imgToAnns.keys())
    #image_ids = coco_gt.getImgIds()
    dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,image_ids,
                                transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
    #-------------------------------------------------------#
    #   获得测试用的图片路径和标签
    #   默认指向根目录下面的datasets/coco文件夹
    #-------------------------------------------------------#
    Image_dir       = "./data/coco/images"
    Json_path       = "./data/coco/annotations/instances_val2017.json"
    map_out_path    = 'map_out'
    test_coco       = COCO(Json_path)
    class_names, _  = get_classes(args.classes_path)
    COCO_LABEL_MAP  = get_coco_label_map(test_coco, class_names)
    
    if not args.display and not args.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')
    
    if args.image is not None:
       if ':' in args.image:
           inp, out = args.image.split(':')
           evalimage(stream_manager_api, inp, out)
       else:
           evalimage(stream_manager_api, args.image)
       return
   
   
#############################################3
    if not osp.exists(map_out_path):
        os.makedirs(map_out_path)
    dataset_size = len(dataset)
    frame_times = MovingAverage()
    progress_bar = ProgressBar(30, dataset_size)
    #image_ids=image_ids[4930:]
    print("Get predict result.")
    make_json   = Make_json(map_out_path, COCO_LABEL_MAP)
    
    for image_idx, image_id in enumerate(image_ids):
        print('image_idx = %d image_id = %d.' % (image_idx, image_id))
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = os.path.join(IMAGEFOLDER, image_info['file_name'])
        
######################################### 2th method       
        image       = Image.open(image_path)
        image_shape = np.array(np.shape(image)[0:2])
        image           = cvtColor(image)
        image_data      = resize_image(image, (544, 544))
        image_data      = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        
#########################################           
        print('Detect image: ', image_idx, ': ', image_info['file_name'],
              ', image id: ', image_id)
        if os.path.exists(image_path) != 1:
            print("The test image does not exist. Exit.")
            exit()
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
        #batch = img[None]
        batch = image_data  ############ method 2    
        batch = batch.astype(np.float32)
        data_input = MxDataInput()
        stream_name=b'im_yolact'
        in_plugin_id=0

        if not send_source_data(0, batch, stream_name, stream_manager_api):
            return
        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        infer_results = stream_manager_api.GetProtobuf(stream_name, in_plugin_id, keyVec)
        if infer_results.size() == 0 or infer_results.size() == 0:
            print("infer_result is null")
            exit()
        if infer_results[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_results[0].errorCode, infer_results[0].data.decode()))
            exit()
        resultList = MxpiDataType.MxpiTensorPackageList()
        resultList.ParseFromString(infer_results[0].messageBuf)
        pred_boxes = np.array(
            [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)]).reshape(1, 18525, 4)
        pred_classes = np.array(
            [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.float32)]).reshape(1, 18525 ,81)
        pred_masks = np.array(
            [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[2].dataStr, dtype=np.float32)]).reshape(1, 18525, 32)
        pred_proto = np.array(
            [np.frombuffer(resultList.tensorPackageVec[0].tensorVec[3].dataStr, dtype=np.float32)]).reshape(1, 136, 136, 32)  
        ######################   
        anchor = get_anchors([544, 544], [24, 48, 96, 192, 384])
        detect = BBoxUtility()
        outputs = tuple([pred_boxes, pred_classes, pred_masks, pred_proto])
        results = detect.decode_nms(outputs, anchor, args.confidence, args.nms_iou, image_shape, args.traditional_nms)
        if results[0] is None:
            continue
        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = [x for x in results]
        if box_thre is None:
            continue
        prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, image_id, make_json)

    make_json.dump()
    print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')

    bbox_dets = test_coco.loadRes(osp.join(map_out_path, "bbox_detections.json"))
    mask_dets = test_coco.loadRes(osp.join(map_out_path, "mask_detections.json"))
    print('\nEvaluating BBoxes:')
    bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox')
    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()
    print('\nEvaluating Masks:')
    bbox_eval = COCOeval(test_coco, mask_dets, 'segm')
    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()
    exit()
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(stream_manager_api, inp, out)
        else:
            evalimage(stream_manager_api, args.image)
        return

if __name__ == '__main__':
    args = parse_args()
#    if args.config is None:
#        model_path = SavePath.from_str(args.trained_model)
#        # TODO: Bad practice? Probably want to do a name lookup instead.
#        args.config = model_path.model_name + '_config'
#        print('Config not specified. Parsed %s from the file name.\n' % args.config)
#        set_cfg(args.config)

    if not os.path.exists('results'):
      os.makedirs('results')

    if args.resume and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_map(ap_data)
        
#    if args.image is None and args.video is None and args.images is None:
#        dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
#                                transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
#        prep_coco_cats()
#    else:
#        dataset = None 

    dataset = None
    val(args=args)

