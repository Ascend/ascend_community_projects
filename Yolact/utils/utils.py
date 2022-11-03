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