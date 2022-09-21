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
import random
import copy
import argparse
import numpy as np
import cv2


TYPEALL = "w55,i13,pl1,w31,p27,il80,w32,i1,il100,pl90,w3,\
           w34,w66,i16,il50,pr70,w56,il70,pdc,pl10,w39,il90,\
           pl70,pl30,pl55,i3,i11,pl35,pr30,w43,pr0,w41,pb,i9,\
           il110,p20,p11,pr20,w13,w42,pl2,pn,w21,w57,zo,pl60,p10,\
           p23,w68,w15,i2,pg,pl5,pl120,pa,pl110,ps,il60,pr80,w24,\
           pl8,p14,i12,p6,p9,pl100,pr40,w16,w45,i14,p28,pr100,pl4,\
           w30,pl50,p31,i5,i10,i18,p19,i17,i19,p29,w69,p21,ip,p12,\
           pr60,pl25,w46,pnl,pl7,pl20,p30,w10,i6,pr50,w20,w47,p13,\
           pne,w63,p5,w22,pl130,p32,i4,pl3,pl40,pl80,w40,pl15,ipp,p1,i15"
TYPENP = "i15,ipp,w40,pl3,pl130,p32,i6,w10,p30,pl7,pnl,w69,,p29,\
            i19,i17,i18,p31,pl4,pl8,w24,pa,w68,zo,pl2,i9,pr0,w43,\
            pl55,w39,pdc,w56,i16,w66,w31,pl1"
TYPEFEW = "i13,w32,w34,pl10,il90,pl35,w41,w21,p10,w15,i2,pl5,\
           i12,p6,p9,w16,w45,p28,pr100,i10,p12,w46,w20,w47,p13,\
           w63,i4"
TYPEALL = TYPEALL.split(',')
TYPENP = TYPENP.split(',')
TYPEFEW = TYPEFEW.split(',')

Type45 = [i for i in TYPEALL if (i not in TYPENP and i not in TYPEFEW)]


def rectcross_(rec__1, rec__2):
    rect_tmp = [max(rec__1[0], rec__2[0]),
                max(rec__1[1], rec__2[1]),
                min(rec__1[2], rec__2[2]),
                min(rec__1[3], rec__2[3])]
    rect_tmp[2] = max(rect_tmp[2], rect_tmp[0])
    rect_tmp[3] = max(rect_tmp[3], rect_tmp[1])
    return rect_tmp


def rectarea_(rect_tmp_):
    return float(
        max(0.0, (rect_tmp_[2] - rect_tmp_[0]) * (rect_tmp_[3] - rect_tmp_[1])))


def calciou_(rect_1, rect_2):
    cre_tmp = rectcross_(rect_1, rect_2)
    ac_tmp = rectarea_(cre_tmp)
    a1_tmp = rectarea_(rect_1)
    a2_tmp = rectarea_(rect_2)
    return ac_tmp / (a1_tmp + a2_tmp - ac_tmp)


def boxlongsize_(box_tmp):
    return max(
        box_tmp['xmax'] -
        box_tmp['xmin'],
        box_tmp['ymax'] -
        box_tmp['ymin'])


def evalannos_(
        annos_gd,
        annos_rt,
        iou=0.75,
        imgids=None,
        check_type=True,
        types=None,
        minscore=40,
        minboxsize=0,
        maxboxsize=400,
        match_same=True):
    ac_n_tmp_, ac_c_tmp_ = 0, 0
    rc_n_tmp_, rc_c_tmp_ = 0, 0
    if imgids is None:
        imgids = annos_rt['imgs'].keys()
    if types is not None:
        types = {t: 0 for t in types}
    mi_tmp = {"imgs": {}}
    wron_tmp = {"imgs": {}}
    rig_tmp = {"imgs": {}}

    for imgid in imgids:
        v = annos_rt['imgs'][imgid]
        vg = annos_gd['imgs'][imgid]

        def convert(objs): return [[obj['bbox'][key] for key in [
            'xmin', 'ymin', 'xmax', 'ymax']] for obj in objs]
        objs_g_tmp_ = vg["objects"]
        objs_r_tmp_ = v["objects"]
        bg_tmp_ = convert(objs_g_tmp_)
        br_tmp_ = convert(objs_r_tmp_)

        match_g_tmp_ = [-1] * len(bg_tmp_)
        match_r_tmp_ = [-1] * len(br_tmp_)
        if types is not None:
            for i, _ in enumerate(match_g_tmp_):
                if not objs_g_tmp_[i]['category'] in types:
                    match_g_tmp_[i] = -2
            for i, _ in enumerate(match_r_tmp_):
                if not objs_r_tmp_[i]['category'] in types:
                    match_r_tmp_[i] = -2
        for i, _ in enumerate(match_r_tmp_):
            if 'score' in objs_r_tmp_[i] and objs_r_tmp_[
                    i]['score'] < minscore:
                match_r_tmp_[i] = -2
        match_tmp = []
        for i, boxg in enumerate(bg_tmp_):
            for j, boxr in enumerate(br_tmp_):
                if match_g_tmp_[i] == -2 or match_r_tmp_[j] == -2:
                    continue
                if match_same and objs_g_tmp_[
                        i]['category'] != objs_r_tmp_[j]['category']:
                    continue
                tiu_tmp = calciou_(boxg, boxr)
                if tiu_tmp > iou:
                    match_tmp.append((tiu_tmp, i, j))
        match_tmp = sorted(match_tmp, key=lambda x: -x[0])
        for tiu_tmp, i, j in match_tmp:
            if match_g_tmp_[i] == -1 and match_r_tmp_[j] == -1:
                match_g_tmp_[i] = j
                match_r_tmp_[j] = i

        for i, _ in enumerate(match_g_tmp_):
            bize_tmp = boxlongsize_(objs_g_tmp_[i]['bbox'])
            erase = False
            if not (bize_tmp >= minboxsize and bize_tmp < maxboxsize):
                erase = True
            if erase:
                if match_g_tmp_[i] >= 0:
                    match_r_tmp_[match_g_tmp_[i]] = -2
                match_g_tmp_[i] = -2

        for i, _ in enumerate(match_r_tmp_):
            bize_tmp = boxlongsize_(objs_r_tmp_[i]['bbox'])
            if match_r_tmp_[i] != -1:
                continue
            if not (bize_tmp >= minboxsize and bize_tmp < maxboxsize):
                match_r_tmp_[i] = -2
        try:
            mi_tmp["imgs"][imgid] = {"objects": []}
            wron_tmp["imgs"][imgid] = {"objects": []}
            rig_tmp["imgs"][imgid] = {"objects": []}
            miss_objs = mi_tmp["imgs"][imgid]["objects"]
            wrong_objs = wron_tmp["imgs"][imgid]["objects"]
            right_objs = rig_tmp["imgs"][imgid]["objects"]
        except KeyError:
            print('error')
        tt = 0
        for i, _ in enumerate(match_g_tmp_):
            if match_g_tmp_[i] == -1:
                miss_objs.append(objs_g_tmp_[i])
        for i, _ in enumerate(match_r_tmp_):
            if match_r_tmp_[i] == -1:
                obj = copy.deepcopy(objs_r_tmp_[i])
                obj['correct_catelog'] = 'none'
                wrong_objs.append(obj)
            elif match_r_tmp_[i] != -2:
                j = match_r_tmp_[i]
                obj = copy.deepcopy(objs_r_tmp_[i])
                if not check_type or objs_g_tmp_[
                        j]['category'] == objs_r_tmp_[i]['category']:
                    right_objs.append(objs_r_tmp_[i])
                    tt += 1
                else:
                    obj['correct_catelog'] = objs_g_tmp_[j]['category']
                    wrong_objs.append(obj)

        rc_n_tmp_ += len(objs_g_tmp_) - match_g_tmp_.count(-2)
        ac_n_tmp_ += len(objs_r_tmp_) - match_r_tmp_.count(-2)

        ac_c_tmp_ += tt
        rc_c_tmp_ += tt
    report = "accuracy:%s, recall:%s" % (
        1 if ac_n_tmp_ == 0 else ac_c_tmp_ * 1.0 / ac_n_tmp_, 1 if rc_n_tmp_ == 0 else rc_c_tmp_ * 1.0 / rc_n_tmp_)
    return report


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filedir",
        type=str,
        default="./data/detection/data/annotations.json",
        help="path to dataset")
    parser.add_argument(
        "--result_anno_file",
        type=str,
        default="./data/detection/Tinghua100K_result_for_test.json",
        help="path to dataset")
    opt = parser.parse_args()
    annos = json.loads(open(opt.filedir).read())
    results_annos = json.loads(open(opt.result_anno_file).read())
    sm = evalannos_(
        annos,
        results_annos,
        iou=0.5,
        check_type=True,
        types=Type45,
        minboxsize=0,
        maxboxsize=400,
        minscore=40)

    print(sm)
