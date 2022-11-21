# MindXSDK 辅助驾驶

## 1 简介
本开发样例基于MindX SDK实现了车辆、交通指示牌分类、车道线检测，并可视化呈现。

## 2 目录结构
本工程名称为Assisted-driving，工程目录如下图所示：
```
ADAS
|---- config
|   |   |---- class.cfg
|   |   |---- coco.names
|   |   |---- coco2014_minival.txt
|   |   |---- segmengtation.cfg
|   |   |---- yolov3_tf_aipp.cfg
|   |   |---- yolov3_tf_bs1_fp16.cfg
|---- data
|   |---- detection  
|   |   |---- data  
|   |       |---- lmdb 				
|   |       |---- marks				
|   |       |---- other				
|   |       |---- test				
|   |       |---- train				
|   |       |---- annotations.json	
|   |       |---- Tinghua100K_result_for_test.json	
|   |---- road_line                      
|   |   |---- data                      
|   |       |---- testing 					
|   |       |---- training					
|   |       |---- validation				
|   |       |---- config_v1.2.json				
|   |       |---- config_v2.0.json				
|   |       |---- demo.py
|   |       |---- LICENSE	
|   |       |---- README	
|   |---- car                      
|   |   |---- annotations 					
|   |   |---- val2014					
|---- models            
|   |---- class.om  
|   |---- class.onnx  
|   |---- detection.om  
|   |---- detection.onnx  
|   |---- lane_segmentation_448x448.om  
|   |---- lane_segmentation_448x448.onnx  
|   |---- yolov3_tf_aipp.om  
|   |---- yolov3_tf.pb  
|---- pipeline                       
|   |   |---- class.pipeline
|   |   |---- detection.pipeline
|   |   |---- road.pipeline
|   |   |---- yolov3_opencv.pipeline
|---- main_car_test.py
|---- main_car.py
|---- main_deteciton.py
|---- main_deteciton_test.py
|---- main_segmentation.py
|---- plot_utils.py
|---- test_detection.py
|---- test_segmentation.py
|---- test_car.py
|---- README.md  
```
## 3 依赖
| 软件名称 | 版本   |
| :--------: | :------: |
|ubuntu 18.04|18.04.1 LTS   |
|CANN|5.0.4|
|MindX SDK|2.0.4|
|Python| 3.9.2|
|numpy | 1.22.4 |
|opencv_python|4.6.0.66|

- 设置环境变量
```
#执行如下命令
. ${SDK-path}/set_env.sh
. ${ascend_toolkit_path}/set_env.sh
```

请注意MindX SDK使用python版本为3.9.2，如出现无法找到python对应lib库请在root下安装python3.9开发库 
```
apt-get install libpython3.9
conda install -c conda-forge pycocotools
```

## 4 模型转换

**步骤1** 获取原始模型权重(.onnx和.om模型) 
&ensp;[模型下载](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/Assisted_driving/model.zip) 

**步骤2** 将模型权重下载完后解压，移动到所在目录“model/”文件夹中，请参考目录结构。 

**步骤3** .om模型转换  

- 使用ATC将.pb和.onnx文件转成为.om文件

> 模型转换使用了ATC工具，如需更多信息请参考：[ATC工具使用指南-快速入门](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html)  

>车辆检测模型需要从此[链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/c-version/YoloV3_for_TensorFlow/zh/1.6/m/YOLOv3_TensorFlow_1.6_model.zip)进行下载，将 YOLOv3_TensorFlow_1.6_model\single\om中的yolov3_tf_aipp.om复制到model目录下。
```
#进行模型转换

atc --framework=5 --model=./model/class.onnx --output=./model/class --out_nodes="class_output" --insert_op_conf=./config/class.config  --input_format=NCHW --input_shape="class_input:1,3,224,224" --log=debug --soc_version=Ascend310

atc --framework=5 --model=./model/detect.onnx --output=./model/detect --out_nodes="detect_output0;detect_output1;detect_output2" --input_format=NCHW --input_shape="detect_input:1,3,1216,1216" --log=debug --soc_version=Ascend310

atc --framework=3 --model=./model/lane_segmentation_448x448.pb  --output=./model/lane_segmentation_448x448 --input_format=NHWC --input_shape="data:1,448,448,3"  --out_nodes="sigmoid/Sigmoid:0"  --enable_small_channel=1 --insert_op_conf=./config/segmentation.config --soc_version=Ascend310 --log=info

atc --framework=3 --model=./model/yolov3_tf.pb --output=./model/yolov3_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3"  --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0" --insert_op_conf=./config/yolov3_tf_aipp.cfg --log=info 

```
- 执行完模型转换脚本后，若提示如下信息说明模型转换成功。
```
ATC run success, welcome to the next use.
```  

经过上述操作，可以在“项目所在目录/model”的子目录下找到detect.om模型、class.om模型和lane_segmentation_448x448.om模型，模型转换操作已全部完成


## 5 准备

### 5.1 数据

> 本文将所使用到的数据集进行整理，将数据集下载后进行解压并放置到项目目录中，参考上述给出文件目录结构。[交通标志和车道线检测数据集 ](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/Assisted_driving/data.zip)。从[车辆检测数据集](https://pjreddie.com/projects/coco-mirror/)将coco2014验证集和标注文件下载，并解压到./data/car/目录下，具体格式请参考目录结构。


### 5.2 适用场景

项目适用于大部分交通道路场景。

## 6 运行

### 6.1 交通指示牌检测
```
#功能测试  支持jpg,png格式图像
python3.9 main_deteciton.py  --image_folder  <测试图像文件夹路径> --save_image  <测试图像结果保存路径>
# python3.9 main_detection.py  --image_folder  ./test  --save_image  ./result
#执行完成后，结果保存在 result 件中。

#进行检测测试

python3.9 main_detection_test.py

python3.9 test_detection.py

#输出精度为
accuracy : 0.8641904761904762, reca11:0.9242208189049538

#参考精度为
accuracy : 0.8642, reca11:0.9242
```
### 6.2 道路线分割
```
#功能测试  仅支持jpg,jpeg格式图像
python3.9 main_segmentation.py  --image_folder  <测试图像文件夹路径>  --save_image  <测试图像结果保存路径>
# python3.9 main_segmentation.py  --image_folder  ./test  --save_image  ./result
执行完成后，在result文件夹，并保存有相应的结果照片。

#进行分割测试
python3.9 main_segmentation.py  

python3.9 test_segmentation.py

#输出精度为
mIou is : 0.3598664687463788

#参考精度为
mIou is : 0.3610
```
### 6.2 车辆检测
```
#功能测试  支持jpg,jpeg,PNG格式图像
python3.9 main_car.py  --image_folder  <测试图像文件夹路径>  --res_dir_name  <测试图像结果保存路径>
# python3.9 main_car.py  --image_folder  ./test  --res_dir_name  ./result
执行完成后，在result文件夹，并保存有相应的结果照片。

#进行车辆测试，测试使用的数据集为config/coco2014_minival.txt，是参考仓库生成的。
python3.9 main_car_test.py 

python3.9 test_car.py

#原模型精度测试结果如下所示：
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.296
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.129
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.156
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.494

```
> 原模型精度测试代码从[此处](https://www.hiascend.com/zh/software/modelzoo/models/detail/C/210261e64adc42d2b3d84c447844e4c7/1)下载，从该页面点击 下载模型脚本 和 下载模型。在模型脚本文件中YoloV3_for_TensorFlow_1.6_code\infer中有开发Mind_SDK的全部代码，在 infer\utils 有精度测试代码，可根据代码修改相关文件路径进行测试；在infer\data\config中有yolov3的配置文件，在infer\convert中有模型转换代码和aipp配置文件；在infer\sdk中保存有模型推理函数；在YoloV3_for_TensorFlow_1.6_code\data中有相关的数据集配置文件。在模型代码文件 YOLOv3_TensorFlow_1.6_model\single\om已经有转换好的om模型直接使用。

## 7 参考链接
> 车道线检测：(https://github.com/vietanhdev/open-adas/blob/master/docs/open-adas.md)  
> 交通指示牌检测：(https://github.com/yangzhaonan18/yolov3_trafficSign_pytorch)  
