from xmlrpc.client import boolean
from StreamManagerApi import *
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import cv2
import os
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

color2 = [(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class cls_AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # val = val.item()
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
    target_weight = np.ones((80, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints, 96, 72), dtype=np.float32)

    # get target & target_weight
    tmp_size = 3 * 3
    for joint_id in range(num_joints):
        image_size = np.array([288, 384])
        heatmap_size = np.array([72,96])
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
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
        # print("g_x, g_y", g_x, g_y)
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
        # print("img_x, img_y", img_x, img_y)

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            # print("target", target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]])

    # if self.use_different_joints_weight:
    #     target_weight = np.multiply(target_weight, self.joints_weight)

    return target, target_weight

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    print("acc", acc)
    return acc, avg_acc, cnt, pred

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
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

def cls_accuracy(output, cls_target):
    """Computes the precision@k for the specified values of k"""
    
    cls_pre = np.array(np.argmax(output, axis=1))
    #print(cls_pre)

    cls_target = np.array([cls_target])

    flag = (cls_pre == cls_target)
    print("flag", flag)
    res = int(flag.item()) * 100.
    return res

def get_img_metas(file_name):
    img = Image.open(file_name)
    #img = cv2.imread(file_name)
    img_size = img.size

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
        result = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
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
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale
    
def gt_xywh2cs(box):
    x, y, w, h = box[:4]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    aspect_ratio = 288 * 1.0 / 384
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    # assert isinstance(batch_heatmaps, np.ndarray), \
    #     'batch_heatmaps should be numpy.ndarray'
    # assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px] - hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def mask_generate(filter, num_joint):
    class_num = [27, 5, 10, 6, 6, 4, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    idx_num =  [0, 27, 32, 42, 48, 54, 58, 60, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    mask = np.zeros(
        (len(filter), num_joint, 2),
        dtype=np.float32
    )
    for i, index in enumerate(filter):
        for j in range(idx_num[index], idx_num[index + 1]):
            mask[i][j][0] = 1
            mask[i][j][1] = 1
    return mask

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (80, 2)
    for i in range(80):
        x_a, y_a = keypoints[i][0], keypoints[i][1]
        cv2.circle(img, (int(x_a), int(y_a)), 10, color2[i], 10)


def evaluate(preds, output_dir, all_boxes, img_path, flag=0):
    if flag == 0:
        rank = 0
    else:
        rank = 1

    res_folder = os.path.join(output_dir, 'results')
    if not os.path.exists(res_folder):
        try:
            os.makedirs(res_folder)
        except Exception:
            print('Fail to make {}'.format(res_folder))

    res_file = os.path.join(
        res_folder, 'keypoints_{}_results_1005.json'.format('coco'))
    print("res_file", res_file, img_path[0])
    # person x (keypoints)
    _kpts = []
    for idx, kpt in enumerate(preds):
        print("idx", idx)
        _kpts.append({
            'keypoints': kpt,
            'center': all_boxes[idx][0:2],
            'scale': all_boxes[idx][2:4],
            'area': all_boxes[idx][4],
            'score': all_boxes[idx][5],
            'image': int(img_path[idx])
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
    for img in kpts.keys():
        img_kpts = kpts[img]
        print(img_kpts)
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
        print("keep", keep)
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    write_coco_keypoint_results(oks_nmsed_kpts, res_file)
    
    info_str = do_python_keypoint_eval(res_file, res_folder)
    name_value = OrderedDict(info_str)
    return name_value, name_value['AP']

def write_coco_keypoint_results(keypoints, res_file):
    coco = COCO('/home/luoyang3/zyj/HRNET-l/HRNet/data/coco/annotations/person_keypoints_val2017.json')
    cats = [cat['name']
             for cat in coco.loadCats(coco.getCatIds())]
    classes = ['__background__'] + cats
    data_pack = [
        {
            #'cat_id': dict(zip(cats, coco.getCatIds([cls]))),
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
    with open(res_file, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
    try:
        json.load(open(res_file))
    except Exception:
        content = []
        with open(res_file, 'r') as f:
            for line in f:
                content.append(line)
        content[-1] = ']'
        with open(res_file, 'w') as f:
            for c in content:
                f.write(c)

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

        for ipt in range(80):
            key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
            key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
            key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

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
    coco = COCO('/home/luoyang3/zyj/HRNET-l/HRNet/data/coco/annotations/person_keypoints_val2017.json')
    ref_file = '/home/luoyang3/zyj/HRNET-l/HRNet/stage1/mindspore/FasterRCNN_for_MindSpore_1.6_code/gt_keypoints_val2017_results_0.json'
    # another_res_file = '/home/luoyang3/zyj/HRNET-l/HRNet/data/coco/annotations/person_keypoints_val2017.json'
    coco_dt = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

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
    print("len(kpts_db)", len(kpts_db))
    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    print("scores", scores)
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    print("order", order)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
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
        print("dx",dx,"dy,",dy)
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def normalize(data, mean, std):
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    if mean.ndim == 1:
        mean = np.reshape(mean, (-1, 1, 1))
    if std.ndim == 1:
        std = np.reshape(std, (-1, 1, 1))
    # _max = np.max(abs(data))
    _div = np.divide(data, 255)  
    _div = np.transpose(_div, (2, 0, 1))
    _sub = np.subtract(_div, mean)  
    arrays = np.divide(_sub, std)  
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
    with open("./pipeline/model1.pipeline", 'rb') as f2:
        pipelineStr2 = f2.read()
    ret2 = streamManagerApi.CreateMultipleStreams(pipelineStr2)
    if ret2 != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()    

    # Construct the input of the stream & check the input image
    dataInput = MxDataInput()
    image_folder = '/home/luoyang3/zyj/HRNET-l/HRNet/data/coco/images/val2017/val2017/val2017/'
    annotation_file = '/home/luoyang3/zyj/HRNET-l/HRNet/data/coco/annotations/person_keypoints_val2017.json'
    annotations = codecs.open(annotation_file, 'r', 'gbk')
    annotations = json.load(annotations)
    image_list = annotations['images']
    cls_target = []

    top1 = cls_AverageMeter()
    kp_acc = AverageMeter()
    coco = COCO(annotation_file)

    # txt为cls 分类情况
    txt = '/home/luoyang3/zyj/HRNET-l/HRNet/stage1/mindspore/FasterRCNN_for_MindSpore_1.6_code/evaluate_result_0922_3.txt'

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
        image_path = os.path.join(image_folder, image_info['file_name'])
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
            center_gt = np.zeros((1,2),dtype=np.float64)
            scale_gt = np.zeros((1,2),dtype=np.float64)
            roi = anns[i]['keypoints'][0]
            img_cls = anns[i]['cls_id']
            bbox_1 = anns[i]['bbox']
            cls_target = [anns[i]['cls_id']]
            bbox_person = (int(bbox_1[0]), int(bbox_1[1])) , (int(bbox_1[2]), int(bbox_1[3]))
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
                    joints_3d[j, 0:2] = affine_transform(joints_3d[j, 0:2], trans_gt)
            
            # double check this all_boxes parts
            all_boxes[idx, 0] = center_gt[0]
            all_boxes[idx, 1] = center_gt[1]
            all_boxes[idx, 2] = scale_gt[0]
            all_boxes[idx, 3] = scale_gt[1]
            all_boxes[idx:idx + 1, 4] = np.prod(scale_gt*200, 0)
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
        uniqueId = streamManagerApi.SendData(streamName1, inPluginId, dataInput)

        #send image data
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
        uniqueId1 = streamManagerApi.SendProtobuf(streamName1, b'appsrc1', protobufVec)

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
        pre_mask = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[2].dataStr, dtype = boolean).reshape((1,80000,1))
        pre_label = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype = np.uint32).reshape((1,80000,1))
        pre_bbox = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float16).reshape((1,80000,5))

        bbox_squee = np.squeeze(pre_bbox.reshape(80000, 5))
        label_squee = np.squeeze(pre_label.reshape(80000, 1))
        mask_squee = np.squeeze(pre_mask.reshape(80000, 1))

        all_bboxes_tmp_mask = bbox_squee[mask_squee, :]
        all_labels_tmp_mask = label_squee[mask_squee]

        if all_bboxes_tmp_mask.shape[0] > 128:
            inds = np.argsort(-all_bboxes_tmp_mask[:, -1]) # 返回降序排列索引值
            inds = inds[:128]
            all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
            all_labels_tmp_mask = all_labels_tmp_mask[inds]

        outputs = []
        outputs_tmp, out_person = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, 81)
        outputs.append(outputs_tmp)

        img = cv2.imread(image_path)
        box_person = (int(out_person[0][0]), int(out_person[0][1])) , (int(out_person[0][2]), int(out_person[0][3]))

        image_bgr = cv2.imread(image_path)
        image = image_bgr[:, :, [2, 1, 0]]

        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = np.transpose(img / 255., (2, 0, 1))
        input.append(img_tensor)

        center, scale = box_to_center_scale(box_person, 288, 384) # 获得center scale
        image_pose = image.copy() 
        
        # model 2 preprocess
        trans = get_affine_transform(center, scale, 0, [288, 384])
        model_input = cv2.warpAffine(
            image,
            trans,
            (288, 384),
            flags = cv2.INTER_LINEAR)

        mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype = np.float32)
        data_test = normalize(model_input, mean, std)
        data_test = data_test.astype('float32')
        data_test = np.reshape(data_test, (1,3,384,288))

        tensors = data_test
        streamName2 = b'test_model2'
        inPluginId = 0

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

        key = "appsrc0".encode('utf-8')
        protobufVec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensorPackageList.SerializeToString()
        protobufVec.push_back(protobuf)

        ret = streamManagerApi.SendProtobuf(streamName2, inPluginId, protobufVec)
        if ret != 0:
            print("Failed to send data to stream.")
            exit()

        keyVec = StringVector()
        keys2 = b"mxpi_tensorinfer0"
        for key in keys2:
            keyVec.push_back(keys)
        infer_result = streamManagerApi.GetProtobuf(streamName2, 0, keyVec)
        tensorList = MxpiDataType.MxpiTensorPackageList()
        tensorList.ParseFromString(infer_result[0].messageBuf)
        keypoint_outputs = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float16).reshape((1,80,96,72))
        cls_outputs = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype = np.float16).reshape((1,23))
    
        #post_process
        if isinstance(keypoint_outputs, list):
            kp_output = keypoint_outputs[-1]
        else:
            kp_output = keypoint_outputs
        
        preds, maxvals = get_final_preds(
                kp_output,
                np.asarray([center]),
                np.asarray([scale]))
        filter = np.argmax(cls_outputs, axis = 1)
        mask = mask_generate(filter, 80)
        
        preds_mask = preds * mask     
        target = np.zeros((80, 96, 72), dtype=np.float32)
        target_weight = np.ones((80, 1), dtype=np.float32)

        target, target_weight = generate_target(joints_3d, joints_3d_vis)
        target_1 = np.expand_dims(target, axis = 0)

        num_images = 3651
        # measure accuracy and record loss
        _, avg_acc, cnt, pred = accuracy(keypoint_outputs, target_1)
        kp_acc.update(avg_acc, cnt)

        prec1 = cls_accuracy(cls_outputs, cls_target)
        print(prec1)
        top1.update(prec1, 1)

        # has mask
        all_preds[idx:idx + num_images, :, 0:2] = preds_mask[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        
        idx += 1
        
        msg = 'Test: [{0}/{1}]\t' \
                'kp_Accuracy {kp_acc.val:.3f} ({kp_acc.avg:.3f})\t' \
                'cls_Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    image_idx+1, 3651, kp_acc=kp_acc, top1=top1)
        print(msg)

        # 保存结果到txt文件中
        pose_filter = np.argmax(cls_outputs, axis = 1)
        with open(txt, "a") as f:
            p_target = cls_target
            pose_out = pose_filter.tolist()
            cc = image_path + ' ' + str(p_target) + ' ' + str(pose_out[i]) + '\n'
            f.write(cc)
        f.close()

    print('cls_Accuracy:{}'.format(top1.avg))
    
    output_dir = './output'
    name_values, perf_indicator = evaluate(all_preds, output_dir, all_boxes, image_path_all, 0)

    model_name = 'pose_hrnet'
    if isinstance(name_values, list):
        for name_value in name_values:
            print(name_value, model_name)
    else:
        print(name_values, model_name) 

    print(perf_indicator)
