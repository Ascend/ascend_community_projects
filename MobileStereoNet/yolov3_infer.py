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

from base64 import decode
import numpy as np
import cv2
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor, post, BTensor

DEVICE_ID = 0  
MODEL_PATH = "./model/yolov3_tf_aipp.om"  
CONFIG_PATH = "./model/yolov3_tf_bs1_fp16.cfg"  
LABEL_PATH = "./model/yolov3.names"  
B_USEDVPP = False  # Enabled with dvpp image processor, false with opencv


def yolo_infer(image_path, yolo_resize):
    yolo = Model(MODEL_PATH, DEVICE_ID)  
    image_tensor = []
    if B_USEDVPP:
        image_processor0 = ImageProcessor(DEVICE_ID)
        decode_img = image_processor0.decode(image_path, base.nv12)
        
        horigin = decode_img.original_height
        worigin = decode_img.original_width

        image_processor1 = ImageProcessor(DEVICE_ID)
        size_cof = Size(yolo_resize, yolo_resize)
        resize_img = image_processor1.resize(decode_img, size_cof)  
        image_tensor = [resize_img.to_tensor()]  

    else:
        image = np.array(cv2.imread(image_path))
        size_cof = (yolo_resize, yolo_resize)

        horigin = image.shape[0]
        worigin = image.shape[1]

        resize_img = cv2.resize(image, size_cof, interpolation=cv2.INTER_LINEAR)
        
        yuv_img = resize_img[np.newaxis, :, :]
        image_tensor = Tensor(yuv_img) 
        
        image_tensor.to_device(DEVICE_ID) 
        image_tensor = [image_tensor]
    outputs = yolo.infer(image_tensor)

    yolov3_post = post.Yolov3PostProcess(
        config_path=CONFIG_PATH, label_path=LABEL_PATH)  
    
    resize_info = base.ResizedImageInfo()
    resize_info.heightResize = yolo_resize
    resize_info.widthResize = yolo_resize
    resize_info.heightOriginal = horigin
    resize_info.widthOriginal = worigin

    inputs = []
    for output in outputs:
        output.to_host()
        n = np.array(output)
        tensor = BTensor(n)  
        inputs.append(base.batch([tensor] * 2, keep_dims=True))

    results = yolov3_post.process(inputs, [resize_info] * 2)
    
    return results[0]
