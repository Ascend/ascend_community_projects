# 行李箱安检
## 1 介绍

本项目利用 YOLOX 目标检测框架，对安检X光图像中的不同类目标进行检测，将检测得到的不同类的目标用不同颜色的矩形框标记。输入一幅图像，可以检测得到图像中大部分类别目标的位置。本方案使用在 SDANet and PIDray dataset 数据集上训练得到的 YOLOX 模型进行目标检测，数据集中共包含 12 个目标类，可以对警棍、老虎钳、榔头、充电宝、剪刀、扳手、枪支、子弹、喷罐、小刀、打火机等目标进行检测。

### 1.1 支持的产品

本项目以昇腾Atlas310卡、Atlas 200DK为主要的硬件平台。

### 1.2 支持的版本

支持的SDK版本为 2.0.4, CANN 版本为 5.0.4。


### 1.3 软件方案介绍

整体业务流程为：待检测图片通过 appsrc 插件输入，然后使用图像解码插件 imagedecoder 对图片进行解码，再通过图像缩放插件 imageresize 将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件 tensorinfer 得到推理结果，推理结果输入 objectpostprocessor 插件进行后处理，得到输入图片中所有的目标框位置和对应的置信度。最后通过输出插件 appsink 获取检测结果，并在外部进行可视化，将检测结果标记到原图上，本系统的各模块及功能描述如表1所示：

表1 系统方案各模块功能描述：

| 序号 | 子系统 | 功能描述     |
| ---- | ------ | ------------ |
| 1    | 图片输入    | 获取 jpg 格式输入图片 |
| 2    | 图片解码    | 解码图片 |
| 3    | 图片缩放    | 将输入图片放缩到模型指定输入的尺寸大小 |
| 4    | 模型推理    | 对输入张量进行推理 |
| 5    | 目标检测后处理    | 从模型推理结果计算检测框的位置和置信度，并保留置信度大于指定阈值的检测框作为检测结果 |
| 6    | 结果输出    | 获取检测结果|
| 7    | 结果可视化    | 将检测结果标注在输入图片上|


### 1.4 代码目录结构与说明

本工程名称为 Luggage-Checker，工程目录如下所示：
```
.
├── build.sh
├── images
│   ├── DetectionPipeline.png
│   ├── EvaluateInfo.png
│   ├── EvaluateInfoPrevious.png
│   ├── warning.png
│   └── VersionError.png
├── postprocess
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── YoloxPostProcess.cpp
│   └── YoloxPostProcess.h
├── python
│   ├── Main
│   │   ├── eval_pre_post.py
│   │   ├── eval_nopre_post.py
│   │   ├── pre_post.py
│   │   ├── nopre_post.py
│   │   ├── visualize.py
│   │   └── preprocess.py
│   ├── models                      # 下载的onnx模型存放在该文件夹下
│   │   ├── aipp-configs
│   │   │   └── yolox_bgr.cfg                  
│   │   ├── yolox_eval.cfg
│   │   └── coco.names                    
│   ├── test    
│   │   ├── data   
│   │   ├── parse_coco.py                      
│   │   └── map_calculate.py                  
│   ├── test_img
│   │   └── test.jpg                        # 需要用户自行添加测试数据
│   └── pipeline
│       └── pre_post.pipeline
└── README.md

```
onnx模型下载[地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects//best.onnx)

om模型下载[地址](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects//yolox_pre_post.om)

### 1.5 技术实现流程图



