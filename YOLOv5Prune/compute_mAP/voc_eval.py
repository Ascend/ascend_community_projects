"""
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
"""

import xml.etree.ElementTree as ET
import os
import _pickle as pickel
import numpy as np


def parse_xml(filename):
    """ Parse xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        object_info = {}
        bbox = obj.find('bndbox')
        object_info['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        object_info['name'] = obj.find('name').text
        object_info['difficult'] = int(obj.find('difficult').text)
        objects.append(object_info)

    return objects


def get_ap(rec, prec):
    """ 
    ap = ap(rec, prec)
    Compute VOC AP given precision and recall.
    """
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    return np.sum(mpre[i + 1] * (mrec[i + 1] - mrec[i]))


def evaluate(detpath,
        annopath,
        imageset,
        classname,
        cachedir,
        ovthresh=0.5):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    with open(imageset, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if os.path.isfile(cachefile):
        print('cachefile = ', cachefile)
        with open(cachefile, 'rb') as f:
            recs = pickel.load(f)
    else:
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_xml(annopath.format(imagename))
        os.system('touch '+cachefile)
        with open(cachefile, 'rb+') as f:
            pickel.dump(recs, f)

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        recall = [obj for obj in recs.get(imagename) if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in recall])
        difficult = np.array([x['difficult'] for x in recall]).astype(np.bool)
        det = [False] * len(recall)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        npos = npos + sum(~difficult)

    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bounding_box = np.array([[float(z) for z in x[2:]] for x in splitlines])

    ind = np.argsort(-confidence)
    scores = np.sort(-confidence)
    bounding_box = bounding_box[ind, :]
    image_ids = [image_ids[x] for x in ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        recall = class_recs.get(image_ids[d])
        bb = bounding_box[d, :].astype(float)
        ovmax = -np.inf
        bounding_box_gt = recall['bbox'].astype(float)

        if bounding_box_gt.size > 0:
            ixmin = np.maximum(bounding_box_gt[:, 0], bb[0])
            iymin = np.maximum(bounding_box_gt[:, 1], bb[1])
            ixmax = np.minimum(bounding_box_gt[:, 2], bb[2])
            iymax = np.minimum(bounding_box_gt[:, 3], bb[3])
            inters = np.maximum(ixmax - ixmin + 1., 0.) * np.maximum(iymax - iymin + 1., 0.)

            union = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bounding_box_gt[:, 2] - bounding_box_gt[:, 0] + 1.) *
                   (bounding_box_gt[:, 3] - bounding_box_gt[:, 1] + 1.) - inters)

            overlaps = inters / union
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax <= ovthresh:
            fp[d] = 1.
        else:            
            if not recall['difficult'][jmax]:
                if not recall['det'][jmax]:
                    tp[d] = 1.
                    recall['det'][jmax] = 1
                else:
                    fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = get_ap(rec, prec)

    return rec, prec, ap