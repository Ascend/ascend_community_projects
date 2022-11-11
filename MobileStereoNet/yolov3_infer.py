from base64 import decode
import numpy as np
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor, post, BTensor
import cv2

device_id = 0  # 芯片ID
model_path = "./model/yolov3_tf_bs1_fp16.om"  # 模型的路径
config_path = "./model/yolov3_tf_bs1_fp16.cfg"  # 模型配置文件的路径
label_path = "./model/yolov3.names"  # 分类标签文件的路径
yolo_resizelen = 416

def yolov3(image_path):
    yolov3 = Model(model_path, device_id)  # 创造模型对象
    imageTensor = []

    # 创造图像处理器对象!!!!!使用该方法处理后数据已在device侧
    imageProcessor0 = ImageProcessor(device_id)
    decodedImg = imageProcessor0.decode(image_path, base.nv12)
    imageProcessor1 = ImageProcessor(device_id)
    size_cof = Size(yolo_resizelen, yolo_resizelen)
    resizeImg = imageProcessor1.resize(decodedImg, size_cof)

    imageTensor = [resizeImg.to_tensor()]  # 推理前需要转换为tensor的List，数据已在device侧无需转移
    print("-----------Img to Tensor Sucess!---------------")

    outputs = yolov3.infer(imageTensor)
    print("-----------YOLO Infer Success!----------------")
    yolov3_post = post.Yolov3PostProcess(
        config_path=config_path, label_path=label_path)  # 构造对应的后处理对象
    
    

    resizeInfo = base.ResizedImageInfo()
    resizeInfo.heightResize = yolo_resizelen
    resizeInfo.widthResize = yolo_resizelen
    resizeInfo.heightOriginal = decodedImg.original_height
    resizeInfo.widthOriginal = decodedImg.original_width

    inputs = []
    for i in range(len(outputs)):
        outputs[i].to_host()
        n = np.array(outputs[i])
        tensor = BTensor(n)  # 后处理需要使用baseTensor类型来构建，文档不全
        inputs.append(base.batch([tensor] * 2, keep_dims=True))

    results = yolov3_post.process(inputs, [resizeInfo] * 2)
    
    return results[0]


