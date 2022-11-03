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
from PIL import Image


def cvtcolor(img):
    if len(np.shape(img)) == 3 and np.shape(img)[2] == 3:
        return img 
    else:
        img = img.convert('RGB')
        return img 


def resize_image(img, sz):
    w, h    = sz
    image = img.resize((w, h), Image.BICUBIC)
    return image


def get_classes(path):
    with open(path, encoding='utf-8') as f:
        names = f.readlines()
    names = [c.strip() for c in names]
    return names, len(names)


def preprocess_input(img):
    m    = (123.68, 116.78, 103.94)
    s     = (58.40, 57.12, 57.38)
    img   = (img - m)/s
    return img


def get_coco_label_map(coco, names):
    label_map = {}

    cat_index_map = {}
    for index, cat in coco.cats.items():
        if cat['name'] == '_background_':
            continue
        cat_index_map[cat['name']] = index
        
    for index, name in enumerate(names):
        label_map[cat_index_map.get(name)] = index + 1
    return label_map