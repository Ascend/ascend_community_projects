from base64 import decode
import numpy as np
from mindx.sdk import base
from mindx.sdk.base import Tensor, Model, Size, log, ImageProcessor, post, BTensor
import cv2

device_id = 0  # 芯片ID
model_path = "./model/yolov3_tf_aipp.om"  # 模型的路径
config_path = "./model/yolov3_tf_bs1_fp16.cfg"  # 模型配置文件的路径
label_path = "./model/yolov3.names"  # 分类标签文件的路径
yolo_resizelen = 416
b_usedvpp = False  # 使用dvpp图像处理器启用，使用opencv时False

def yolov3(image_path):
    yolov3 = Model(model_path, device_id)  # 创造模型对象
    imageTensor = []
    if b_usedvpp:
    # 创造图像处理器对象!!!!!使用该方法处理后数据已在device侧
        imageProcessor0 = ImageProcessor(device_id)
        decodedImg = imageProcessor0.decode(image_path, base.nv12)
        
        horigin = decodedImg.original_height
        worigin = decodedImg.original_width

        imageProcessor1 = ImageProcessor(device_id)
        size_cof = Size(yolo_resizelen, yolo_resizelen)
        resizeImg = imageProcessor1.resize(decodedImg, size_cof)
        imageTensor = [resizeImg.to_tensor()]  # 推理前需要转换为tensor的List，数据已在device侧无需转移

    else:
        image = np.array(cv2.imread(image_path))
        size_cof = (yolo_resizelen, yolo_resizelen)

        horigin = image.shape[0]
        worigin = image.shape[1]

        resizeImg = cv2.resize(image, size_cof, interpolation=cv2.INTER_LINEAR)
        
        yuv_img =  cv2.cvtColor(resizeImg, cv2.COLOR_BGR2YUV)
        yuv_img = yuv_img[np.newaxis, :, :]
        imageTensor = Tensor(yuv_img) # 推理前需要转换为tensor的List，使用Tensor类来构建。
        
        imageTensor.to_device(device_id) # !!!!!重要，需要转移至device侧
        imageTensor = [imageTensor]
    outputs = yolov3.infer(imageTensor)
    print("-----------YOLO Infer Success!----------------")
    yolov3_post = post.Yolov3PostProcess(
        config_path=config_path, label_path=label_path)  # 构造对应的后处理对象
    
    resizeInfo = base.ResizedImageInfo()
    resizeInfo.heightResize = yolo_resizelen
    resizeInfo.widthResize = yolo_resizelen
    resizeInfo.heightOriginal = horigin
    resizeInfo.widthOriginal = worigin

    inputs = []
    for i in range(len(outputs)):
        outputs[i].to_host()
        n = np.array(outputs[i])
        tensor = BTensor(n)  # 后处理需要使用baseTensor类型来构建，文档不全
        inputs.append(base.batch([tensor] * 2, keep_dims=True))

    results = yolov3_post.process(inputs, [resizeInfo] * 2)
    
    return results[0]
