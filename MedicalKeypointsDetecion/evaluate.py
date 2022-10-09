from xmlrpc.client import boolean
import numpy as np
import cv2
import os
import stat
import math
from PIL import Image
import mindspore as ms
import mindspore.ops as ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import codecs
from collections import defaultdict
from collections import OrderedDict
from StreamManagerApi import *
import MxpiDataType_pb2 as MxpiDataType

color2 = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255),
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), 
          (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 139), (0, 69, 255), ]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class ClassAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_target(joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = 80
    _target_weight = np.ones((80, 1), dtype=np.float32)
    _target_weight[:, 0] = joints_vis[:, 0]

    _target = np.zeros((num_joints, 96, 72), dtype=np.float32)

    # get target & target_weight
    tmp_size = 3 * 3
    for joint_id in range(num_joints):
        image_size = np.array([288, 384])
        heatmap_size = np.array([72, 96])
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            _target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 3 ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = _target_weight[joint_id]
        if v > 0.5:
            _target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return _target, _target_weight


def accuracy(output, _target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    _idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        _pred, _ = get_max_preds(output)
        _target, _ = get_max_preds(_target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((_pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(_pred, _target, norm)

    acc = np.zeros((len(_idx) + 1))
    avg_accuracy = 0
    count = 0
    idx_length = len(_idx)
    for ii in range(idx_length):
        acc[ii + 1] = dist_acc(dists[_idx[ii]])
        if acc[ii + 1] >= 0:
            avg_accuracy = avg_accuracy + acc[ii + 1]
            count += 1

    avg_accuracy = avg_accuracy / count if count != 0 else 0
    if count != 0:
        acc[0] = avg_accuracy
    return avg_accuracy, count, _pred


def calc_dists(_preds, _target, _normalize):
    _preds = _preds.astype(np.float32)
    _target = _target.astype(np.float32)
    dists = np.zeros((_preds.shape[1], _preds.shape[0]))
    for n in range(_preds.shape[0]):
        for c in range(_preds.shape[1]):
            if _target[n, c, 0] > 1 and _target[n, c, 1] > 1:
                normed_preds = _preds[n, c, :] / _normalize[n]
                normed_targets = _target[n, c, :] / _normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def cls_accuracy(output, _cls_target):
    """Computes the precision@k for the specified values of k"""
    cls_pre = np.array(np.argmax(output, axis=1))
    _cls_target = np.array([_cls_target])
    flag = (cls_pre == _cls_target)
    res = int(flag.item()) * 100.
    return res


def get_img_metas(file_name):
    _img = Image.open(file_name)
    img_size = _img.size

    org_width, org_height = img_size
    resize_ratio = 1280 / org_width
    if resize_ratio > 768 / org_height:
        resize_ratio = 768 / org_height

    img_metas = np.array([img_size[1], img_size[0]] +
                         [resize_ratio, resize_ratio])
    return img_metas


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32)
                  for i in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        result_person = bboxes[labels == 0, :]
    return result, result_person


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center_temp = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center_temp[0] = bottom_left_x + box_width * 0.5
    center_temp[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale_temp = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center_temp[0] != -1:
        scale_temp = scale_temp * 1.25

    return center_temp, scale_temp


def gt_xywh2cs(box):
    x, y, w, h = box[:4]
    _center = np.zeros((2), dtype=np.float32)
    _center[0] = x + w * 0.5
    _center[1] = y + h * 0.5
    aspect_ratio = 288 * 1.0 / 384
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    _scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if _center[0] != -1:
        _scale = _scale * 1.25

    return _center, _scale


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def transform_preds(coords, c, s, output_size):
    target_coords = np.zeros(coords.shape)
    trans_temp = get_affine_transform(c, s, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans_temp)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
        center_temp, _scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(_scale, np.ndarray) and not isinstance(_scale, list):
        _scale = np.array([_scale, _scale])

    scale_tmp = _scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center_temp + scale_tmp * shift
    src[1, :] = center_temp + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        _trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        _trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return _trans


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    _maxvals = np.amax(heatmaps_reshaped, 2)
    _maxvals = _maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    _preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    _preds[:, :, 0] = (_preds[:, :, 0]) % width
    _preds[:, :, 1] = np.floor((_preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(_maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    _preds *= pred_mask
    return _preds, _maxvals


def get_final_preds(batch_heatmaps, c, s):
    coords, _maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    _preds = coords.copy()
    # Transform back
    for ii in range(coords.shape[0]):
        _preds[ii] = transform_preds(
            coords[ii], c[ii], s[ii], [heatmap_width, heatmap_height]
        )
    return _preds, _maxvals


def mask_generate(_filter, num_joint):
    class_num = [27, 5, 10, 6, 6, 4, 2, 3, 3, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    idx_num = [0, 27, 32, 42, 48, 54, 58, 60, 63, 66, 67,
               68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    _mask = np.zeros(
        (len(_filter), num_joint, 2),
        dtype=np.float32
    )
    for i1, index in enumerate(_filter):
        for j1 in range(idx_num[index], idx_num[index + 1]):
            _mask[i1][j1][0] = 1
            _mask[i1][j1][1] = 1

    return _mask


def evaluate(preds, _output_dir, _all_boxes, img_path, flag=0):
    if flag == 0:
        rank = 0
    else:
        rank = 1

    res_folder = os.path.join(_output_dir, 'results')
    if not os.path.exists(res_folder):
        try:
            os.makedirs(res_folder)
        except Exception:
            print('Fail to make {}'.format(res_folder))

    res_file = os.path.join(
        res_folder, 'keypoints_{}_results_0.json'.format('coco'))
    _kpts = []
    for _idx, kpt in enumerate(preds):
        print("idx", _idx)
        _kpts.append({
            'keypoints': kpt,
            'center': _all_boxes[_idx][0:2],
            'scale': _all_boxes[_idx][2:4],
            'area': _all_boxes[_idx][4],
            'score': _all_boxes[_idx][5],
            'image': int(img_path[_idx])
        })
    # image x person x (keypoints)
    kpts = defaultdict(list)
    for kpt in _kpts:
        kpts[kpt['image']].append(kpt)

    # rescoring and oks nms
    num_joints = 80
    in_vis_thre = 0.2
    oks_thre = 0.9
    oks_nmsed_kpts = []
    for _img in kpts.keys():
        img_kpts = kpts[_img]
        for n_p in img_kpts:
            box_score = n_p['score']
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, num_joints):
                t_s = n_p['keypoints'][n_jt][2]
                if t_s > in_vis_thre:
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            # rescoring
            n_p['score'] = kpt_score * box_score

        keep = oks_nms(
            [img_kpts[i] for i in range(len(img_kpts))],
            oks_thre
        )
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    write_coco_keypoint_results(oks_nmsed_kpts, res_file)

    info_str = do_python_keypoint_eval(res_file, res_folder)
    _name_value = OrderedDict(info_str)
    return _name_value, _name_value['AP']


def write_coco_keypoint_results(keypoints, res_file):
    coco_anno = COCO('./data/person_keypoints_val2017.json') #修改为测试集的标注json文件路径
    cats = [cat['name']
            for cat in coco_anno.loadCats(coco_anno.getCatIds())]
    classes = ['__background__'] + cats
    data_pack = [
        {
            'cat_id': 1,
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        }
        for cls_ind, cls in enumerate(classes) if not cls == '__background__'
    ]

    results = coco_keypoint_results_one_category_kernel(data_pack[0])
    print('=> writing results json to %s' % res_file)
    FLAGS = os.O_WRONLY | os.O_CREAT
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(res_file, FLAGS, MODES), "w") as f1:
        json.dump(results, f1, sort_keys=True, indent=4)
    try:
        json.load(open(res_file))
    except Exception:
        content = []
        with open(res_file, 'r') as f2:
            for line in f2:
                content.append(line)
        content[-1] = ']'
        with os.fdopen(os.open(res_file, FLAGS, MODES), "w") as f3:
            for c in content:
                f3.write(c)


def coco_keypoint_results_one_category_kernel(data_pack):
    cat_id = data_pack['cat_id']
    keypoints = data_pack['keypoints']
    cat_results = []

    for img_kpts in keypoints:
        if len(img_kpts) == 0:
            continue

        _key_points = np.array([img_kpts[k]['keypoints']
                                for k in range(len(img_kpts))])
        key_points = np.zeros(
            (_key_points.shape[0], 80 * 3), dtype=np.float64
        )

        for _ipt in range(80):
            key_points[:, _ipt * 3 + 0] = _key_points[:, _ipt, 0]
            key_points[:, _ipt * 3 + 1] = _key_points[:, _ipt, 1]
            # keypoints score.
            key_points[:, _ipt * 3 + 2] = _key_points[:, _ipt, 2]

        result = [
            {
                'image_id': img_kpts[k]['image'],
                'category_id': cat_id,
                'keypoints': list(key_points[k]),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            }
            for k in range(len(img_kpts))
        ]
        cat_results.extend(result)

    return cat_results


def do_python_keypoint_eval(res_file, res_folder):
    coco_anno = COCO('./data/person_keypoints_val2017.json')
    coco_dt = coco_anno.loadRes(res_file)
    coco_eval = COCOeval(coco_anno, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats_names = [
        'AP',
        'Ap .5',
        'AP .75',
        'AP (M)',
        'AP (L)',
        'AR',
        'AR .5',
        'AR .75',
        'AR (M)',
        'AR (L)']
    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))
    return info_str


def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(kpts_db) == 0:
        print("len is 0")
        return []
    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten()
                    for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        _i = order[0]
        keep.append(_i)
        oks_ovr = oks_iou(kpts[_i],
                          kpts[order[1:]],
                          areas[_i],
                          areas[order[1:]],
                          sigmas,
                          in_vis_thre)
        _inds = np.where(oks_ovr <= thresh)[0]
        order = order[_inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                          1.07, 1.07, .87, .87, .89, .89]) / 10.0
    _vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / _vars / \
            ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def normalize(data, _mean, _std):
    if not isinstance(_mean, np.ndarray):
        _mean = np.array(_mean)
    if not isinstance(_std, np.ndarray):
        _std = np.array(_std)
    if _mean.ndim == 1:
        _mean = np.reshape(_mean, (-1, 1, 1))
    if _std.ndim == 1:
        _std = np.reshape(_std, (-1, 1, 1))
    _div = np.divide(data, 255)  # i.e. _div = data / _max
    _div = np.transpose(_div, (2, 0, 1))
    _sub = np.subtract(_div, _mean)  # i.e. arrays = _div - mean
    arrays = np.divide(_sub, _std)  # i.e. arrays = (_div - mean) / std
    return arrays


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/model1.pipeline", 'rb') as f1:
        pipelineStr = f1.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    ret2 = streamManagerApi.InitManager()
    if ret2 != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/model2.pipeline", 'rb') as f2:
        pipelineStr2 = f2.read()
    ret2 = streamManagerApi.CreateMultipleStreams(pipelineStr2)
    if ret2 != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream & check the input image
    dataInput = MxDataInput()
    ImageFolder = './data/'
    annotation_file = './data/person_keypoints_val2017.json'
    annotations = codecs.open(annotation_file, 'r', 'gbk')
    annotations = json.load(annotations)
    image_list = annotations['images']
    cls_target = []

    top1 = ClassAverageMeter()
    kp_acc = AverageMeter()
    coco = COCO(annotation_file)

    # txt为cls 分类情况,在该路径下创建新txt并修改该路径
    TXT = 'evaluate_result_0922_3.txt'

    num_samples = 3651
    all_preds_no_mask = np.zeros((num_samples, 80, 3), dtype=np.float32)
    all_preds = np.zeros((num_samples, 80, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    image_path_all = []
    filenames = []
    imgnums = []
    idx = 0

    for image_idx, image_info in enumerate(image_list):
        image_path = os.path.join(ImageFolder, image_info['file_name'])
        image_id = image_info['id']
        width = image_info['width']
        height = image_info['height']

        print(image_id, width, height)
        ann_ids = coco.getAnnIds(image_id)
        anns = coco.loadAnns(ann_ids)
        anns_len = len(anns)
        joints_3d = np.zeros((80, 3), dtype=np.float64)
        joints_3d_vis = np.zeros((80, 3), dtype=np.float64)
        cls_target = 0
        image_path_all.extend([image_id])
        for i, _ in enumerate(range(anns_len)):
            center_gt = np.zeros((1, 2), dtype=np.float64)
            scale_gt = np.zeros((1, 2), dtype=np.float64)
            roi = anns[i]['keypoints'][0]
            img_cls = anns[i]['cls_id']
            bbox_1 = anns[i]['bbox']
            cls_target = [anns[i]['cls_id']]
            bbox_person = (int(bbox_1[0]), int(
                bbox_1[1])), (int(bbox_1[2]), int(bbox_1[3]))
            center_gt, scale_gt = gt_xywh2cs(bbox_1)
            for ipt in range(80):
                joints_3d[ipt, 0] = anns[i]['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = anns[i]['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = anns[i]['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0
            trans_gt = get_affine_transform(center_gt, scale_gt, 0, [288, 384])

            for j in range(80):
                if joints_3d_vis[j, 0] > 0.0:
                    joints_3d[j, 0:2] = affine_transform(
                        joints_3d[j, 0:2], trans_gt)

            # double check this all_boxes parts
            all_boxes[idx, 0] = center_gt[0]
            all_boxes[idx, 1] = center_gt[1]
            all_boxes[idx, 2] = scale_gt[0]
            all_boxes[idx, 3] = scale_gt[1]
            all_boxes[idx:idx + 1, 4] = np.prod(scale_gt * 200, 0)
            all_boxes[idx:idx + 1, 5] = 1

        if os.path.exists(image_path) != 1:
            print("Failed to get the input picture. Please check it!")
            streamManagerApi.DestroyAllStreams()
            exit()

        with open(image_path, 'rb') as f:
            dataInput.data = f.read()
            print("load:", image_path)

        # Inputs data to a specified stream based on streamName.
        streamName1 = b'model1'
        inPluginId = 0
        uniqueId = streamManagerApi.SendData(
            streamName1, inPluginId, dataInput)

        # send image data
        metas = get_img_metas(image_path).astype(np.float32).tobytes()

        key = b'appsrc1'
        visionList = MxpiDataType.MxpiVisionList()
        visionVec = visionList.visionVec.add()
        visionVec.visionData.deviceId = 0
        visionVec.visionData.memType = 0
        visionVec.visionData.dataStr = metas
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiVisionList'
        protobuf.protobuf = visionList.SerializeToString()
        protobufVec = InProtobufVector()
        protobufVec.push_back(protobuf)

        inPluginId1 = 1
        uniqueId1 = streamManagerApi.SendProtobuf(
            streamName1, b'appsrc1', protobufVec)

        if uniqueId1 < 0:
            print("Failed to send data to stream.")
            exit()

        keyVec = StringVector()
        keys = b"mxpi_tensorinfer0"
        for key in keys:
            keyVec.push_back(keys)
        infer_result = streamManagerApi.GetProtobuf(streamName1, 0, keyVec)
        # print the infer result

        if infer_result.size() == 0:
            print("infer_result is null")
            exit()

        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
                infer_result[0].errorCode, infer_result[0].data.decode()))
            exit()

        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[0].messageBuf)
        pre_mask = np.frombuffer(
            tensorList.tensorPackageVec[0].tensorVec[2].dataStr, dtype=boolean).reshape(
            (1, 80000, 1))
        pre_label = np.frombuffer(
            tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype=np.uint32).reshape(
            (1, 80000, 1))
        pre_bbox = np.frombuffer(
            tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16).reshape(
            (1, 80000, 5))

        bbox_squee = np.squeeze(pre_bbox.reshape(80000, 5))
        label_squee = np.squeeze(pre_label.reshape(80000, 1))
        mask_squee = np.squeeze(pre_mask.reshape(80000, 1))

        all_bboxes_tmp_mask = bbox_squee[mask_squee, :]
        all_labels_tmp_mask = label_squee[mask_squee]

        if all_bboxes_tmp_mask.shape[0] > 128:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1])  # 返回降序排列索引值
            inds = inds[:128]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]

        outputs = []
        outputs_tmp, out_person = bbox2result_1image(
            all_bboxes_tmp_mask, all_labels_tmp_mask, 81)
        outputs.append(outputs_tmp)

        img = cv2.imread(image_path)
        box_person = (int(out_person[0][0]), int(out_person[0][1])), (int(
            out_person[0][2]), int(out_person[0][3]))

        image_bgr = cv2.imread(image_path)
        image = image_bgr[:, :, [2, 1, 0]]

        center, scale = box_to_center_scale(
            box_person, 288, 384)  # 获得center scale
        image_pose = image.copy()

        # model 2 preprocess
        trans = get_affine_transform(center, scale, 0, [288, 384])
        model_input = cv2.warpAffine(
            image,
            trans,
            (288, 384),
            flags=cv2.INTER_LINEAR)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        data_test = normalize(model_input, mean, std)
        data_test = data_test.astype('float32')
        data_test = np.reshape(data_test, (1, 3, 384, 288))

        tensors = data_test
        STREAM_NAME_2 = b'model2'

        tensorPackageList = MxpiDataType.MxpiTensorPackageList()
        tensorPackage = tensorPackageList.tensorPackageVec.add()
        print(tensors.shape)
        array_bytes = tensors.tobytes()
        dataInput = MxDataInput()
        dataInput.data = array_bytes
        tensorVec = tensorPackage.tensorVec.add()
        tensorVec.deviceId = 0
        tensorVec.memType = 0
        for i in tensors.shape:
            tensorVec.tensorShape.append(i)
        tensorVec.dataStr = dataInput.data
        tensorVec.tensorDataSize = len(array_bytes)

        KEY_2 = "appsrc0".encode('utf-8')
        protobufVec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = KEY_2
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensorPackageList.SerializeToString()
        protobufVec.push_back(protobuf)

        ret = streamManagerApi.SendProtobuf(
            STREAM_NAME_2, 0, protobufVec)
        if ret != 0:
            print("Failed to send data to stream.")
            exit()

        keyVec = StringVector()
        KEYS_2 = b"mxpi_tensorinfer0"
        for key in KEYS_2:
            keyVec.push_back(KEYS_2)
        infer_result = streamManagerApi.GetProtobuf(STREAM_NAME_2, 0, keyVec)
        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[0].messageBuf)
        keypoint_outputs = np.frombuffer(
            tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float16).reshape(
            (1, 80, 96, 72))
        cls_outputs = np.frombuffer(
            tensorList.tensorPackageVec[0].tensorVec[1].dataStr,
            dtype=np.float16).reshape(
            (1,
             23))

        # post_process
        if isinstance(keypoint_outputs, list):
            kp_output = keypoint_outputs[-1]
        else:
            kp_output = keypoint_outputs

        preds, maxvals = get_final_preds(
            kp_output,
            np.asarray([center]),
            np.asarray([scale]))
        filters = np.argmax(cls_outputs, axis=1)
        mask = mask_generate(filters, 80)

        preds_mask = preds * mask
        target = np.zeros((80, 96, 72), dtype=np.float32)
        target_weight = np.ones((80, 1), dtype=np.float32)

        target, target_weight = generate_target(joints_3d, joints_3d_vis)
        target_1 = np.expand_dims(target, axis=0)

        NUM_IMAGES = 3651
        # measure accuracy and record loss
        avg_acc, cnt, pred = accuracy(keypoint_outputs, target_1)
        kp_acc.update(avg_acc, cnt)

        prec1 = cls_accuracy(cls_outputs, cls_target)
        print(prec1)
        top1.update(prec1, 1)

        # has mask
        all_preds[idx:idx + NUM_IMAGES, :, 0:2] = preds_mask[:, :, 0:2]
        all_preds[idx:idx + NUM_IMAGES, :, 2:3] = maxvals

        idx += 1

        MSG = 'Test: [{0}/{1}]\t' \
            'kp_Accuracy {kp_acc.val:.3f} ({kp_acc.avg:.3f})\t' \
            'cls_Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                image_idx + 1, NUM_IMAGES, kp_acc=kp_acc, top1=top1)
        print(MSG)

        # 保存结果到txt文件中
        pose_filter = np.argmax(cls_outputs, axis=1)
        FLAGS = os.O_WRONLY | os.O_CREAT
        MODES = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(TXT, FLAGS, MODES), "a") as f:
            p_target = cls_target
            pose_out = pose_filter.tolist()
            cc = image_path + ' ' + \
                str(p_target) + ' ' + str(pose_out) + '\n'
            f.write(cc)
        f.close()

    print('cls_Accuracy:{}'.format(top1.avg))

    OUTPUT_DIR = './output'
    name_values, perf_indicator = evaluate(
        all_preds, OUTPUT_DIR, all_boxes, image_path_all, 0)

    MODEL_NAME = 'pose_hrnet'
    if isinstance(name_values, list):
        for name_value in name_values:
            print(name_value, MODEL_NAME)
    else:
        print(name_values, MODEL_NAME)

    print(perf_indicator)
