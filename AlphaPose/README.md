# AlphaPose人体关键点估计

## 1 介绍

AlphaPose 人体关键点估计前、后处理插件基于 MindX SDK 开发，在昇腾芯片上进行人体关键点估计，将结果进行可视化并保存。主要处理流程为：输入视频 >视频解码 >图像缩放 >行人检测模型推理 >行人检测模型后处理 >人体关键点估计模型前处理 >人体关键点估计模型推理  > 人体关键点估计模型后处理 >关键点输出及可视化 >视频编码。

人体关键点检测是指在行人检测的基础上，对检测出来的所有行人进行人体 17 个关键点的检测，包括鼻子、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左手腕、右手腕、左髋、右髋、左膝、右膝、左踝、右踝。然后将关键点正确配对组成相应的人体骨架，展示人体姿态。本方案采取 AlphaPose 人体关键点检测模型，将待检测图片输入模型进行推理，推理得到包含人体 17 个关键点信息的 Heatmaps，再从 Heatmaps 中提取得到人体关键点位置等信息。

### 1.1 支持的产品

Atlas 200DK

### 1.2 支持的版本

本样例配套的 CANN 版本为 [5.0.5alpha001](https://www.hiascend.com/software/cann/community)。支持的 SDK 版本为 [2.0.4](https://www.hiascend.com/software/Mindx-sdk)。

在 Atlas200DK 上进行环境搭建可参考：[开发板环境搭建](https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk开发板环境搭建.md)。

### 1.3 软件方案介绍

基于 MindX SDK 的 AlphaPose 人体关键点估计业务流程为：待检测视频存放在 live555 服务器上经 mxpi_rtspsrc 拉流插件输入，然后使用视频解码插件 mxpi_videodecoder 将视频解码成图片，再通过图像缩放插件 mxpi_imageresize 将图像缩放至满足行人检测模型要求的输入图像大小要求，缩放后的图像输入行人检测模型推理插件 mxpi_tensorinfer0 得到行人检测结果，本项目开发的 AlphaPose 前处理插件处理行人推理结果，得到满足AlphaPose模型要求的输入图像，AlphaPose 前处理后的图像经过 AlphaPose 模型推理插件 mxpi_tensorinfer1 得到包含人体 17 个关键点信息的 Heatmaps，本项目开发的 AlphaPose 后处理插件处理 AlphaPose 模型推理结果，得到人体关键点位置与置信度信息。最后输出插件 appsink 获取 AlphaPose 后处理插件输出结果，并在外部进行人体姿态可视化描绘与视频编码。

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统               | 功能描述                                                     |
| ---- | -------------------- | :----------------------------------------------------------- |
| 1    | 视频输入             | 接收外部调用接口的输入视频路径，对视频进行拉流，并将拉取的裸流存储到缓冲区（buffer）中，并发送到下游插件。 |
| 2    | 视频解码             | 用于视频解码，当前只支持 H264/H265 格式。                    |
| 5    | 图像缩放             | 对解码后的 YUV 格式的图像进行指定宽高的缩放，暂时只支持 YUV 格式的图像。 |
| 6    | 模型推理             | 行人目标检测，目前只支持单 tensor 输入（图像数据）的推理模型。 |
| 7    | 目标检测后处理       | 实现对 yolo 目标检测模型输出的 tensor 解析，获取对于行人的目标检测框。 |
| 8    | 人体关键点估计前处理 | 根据目标检测后处理插件对视频解码后的图像进行仿射变换处理。   |
| 9    | 模型推理             | 人体关键点检测，目前只支持单 tensor 输入（图像数据）的推理模型。 |
| 7    | 人体关键点估计后处理 | 实现对 AlphaPose 模型输出的 tensor 解析，获取人体关键点位置与置信度信息。 |
| 11   | 人体姿态可视化       | 实现对每帧图像标注人体关键点并进行连接。                     |
| 12   | 视频编码             | 将标注后的图片进行视频编码，输出视频。                       |

### 1.4 代码目录结构与说明

本工程名称为 AlphaPose，工程目录如下图所示：

```
.
├── README.md
├── build.sh
├── image
│   ├── acctest.png
│   └── speedtest.png
├── models
│   ├── aipp_192_256_rgb.cfg
│   ├── aipp_yolov3_416_416.aippconfig
│   ├── yolov3.names
│   └── yolov3_tf_bs1_fp16.cfg
├── pipeline
│   ├── evaluate.pipeline
│   ├── image.pipeline
│   └── video.pipeline
├── plugin
│   ├── postprocess
│   │   ├── CMakeLists.txt
│   │   ├── MxpiAlphaposePostProcess.cpp
│   │   ├── MxpiAlphaposePostProcess.h
│   │   └── build.sh
│   └── preprocess
│       ├── CMakeLists.txt
│       ├── MxpiAlphaposePreProcess.cpp
│       ├── MxpiAlphaposePreProcess.h
│       └── build.sh
├── proto
│   ├── CMakeLists.txt
│   ├── build.sh
│   └── mxpiAlphaposeProto.proto
├── run.sh
└── src
    ├── evaluate.py
    ├── image.py
    ├── utils
    │   └── visualization.py
    └── video.py
```



### 1.5 技术实现流程图

![](./image/SDK流程图.png)

AlphaPose模型前处理插件的输入有两个，一个是视频解码插件输出的 YUV 格式的图像帧，一个是检测后处理插件输出的图像帧中人体的位置信息。AlphaPose模型前处理插件整体流程为：

1.  读取视频解码插件输出的 yuv 格式的图像帧数据，并对其进行 YUV 到 RGB 色域的转换。
2.  读取检测后处理插件输出的图像帧中人体的位置信息，根据该位置信息计算人体中心的位置与人体所占面积的宽高。
3.  根据前面两个步骤所获得的信息，对第一步的 RGB 图像进行放射变换。

AlphaPose模型后处理插件的输入也有有两个，一个是检测后处理插件输出的图像帧中人体的位置信息，一个是AlphaPose模型推理插件输出的张量，包含包含图像帧中检测到的所有人体 17 个关键点信息的 Heatmaps。后处理插件的整体流程为：

1.  读取检测后处理插件输出的图像帧中人体的位置信息，根据该位置信息计算人体中心的位置与人体所占面积的宽高。
2.  读取检测AlphaPose模型推理插件输出的包含包含图像帧中检测到的所有人体 17 个关键点信息的 Heatmaps，寻找每张 Heatmap 中的最大值作为该关键点的得分，最大值的位置作为该关键点在 Heatmap 中的位置，然后再结合第一步的信息通过放射变换获取该关键点在原图上的坐标。
3.  进行PoseNMS，通过姿态距离+空间距离作为度量标准，设定阈值，筛选出单一的姿态。



## 2 环境依赖

推荐系统为 ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称            | 版本          |
| ------------------- | ------------- |
| MindX SDK           | 2.0.4         |
| ubuntu              | 18.04.1 LTS   |
| Ascend-CANN-toolkit | 5.0.5alpha001 |
| python              | 3.9.2         |

在编译运行项目前，需要设置环境变量：

```shell
export LD_LIBRARY_PATH=/var/davinci/driver/lib64:/var/davinci/driver/lib64/common:/var/davinci/driver/lib64/driver:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit
export ASCEND_AUTOML_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools

export MX_SDK_HOME=${SDK安装路径}/mxVision
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors":"${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":${LD_LIBRARY_PATH}
export PYTHONPATH=${MX_SDK_HOME}/python:$PYTHONPATH
```

[^注]: 其中 **${SDK安装路径}** 替换为用户的 SDK 安装路径，install_path 替换为开发套件包所在路径。LD_LIBRARY_PATH 用以加载开发套件包中 lib 库。



## 3 软件依赖

推理中涉及到第三方软件依赖如下表所示。

| 依赖软件 | 版本       | 说明                           | 使用教程                                                     |
| -------- | ---------- | ------------------------------ | ------------------------------------------------------------ |
| live555  | 1.09       | 实现视频转rstp进行推流         | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) |
| ffmpeg   | 2021-07-21 | 实现mp4格式视频转为264格式视频 | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md#https://ffmpeg.org/download.html) |



## 4 模型转换

### 4.1 YOLOv3 模型转换

本项目中适用的第一个模型是 YOLOv3 模型，可通过[地址](https://www.hiascend.com/zh/software/modelzoo/detail/1/ba2a4c054a094ef595da288ecbc7d7b4)获取，下载后使用模型转换工具 ATC 将 pb 模型转换为 om 模型，模型转换工具相关介绍参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html 。

模型转换，步骤如下：

1. 将上述 pb 模型下载至 `AlphaPose/models` 文件夹下，文件名为：`yolov3_tf.pb` 。
2. 进入 `AlphaPose/models` 文件夹下执行命令：

```shell
atc --model=./yolov3_tf.pb --framework=3 --output=./yolov3_tf_bs1_fp16 --soc_version=Ascend310 --insert_op_conf=./aipp_yolov3_416_416.aippconfig --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_22/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_6/BiasAdd:0"
```

执行该命令后会在当前文件夹下生成项目需要的模型文件 `yolov3_tf_bs1_fp16.om`。执行后终端输出为：

```shell
ATC run success, welcome to the next use.
```

表示命令执行成功

### 4.2 AlphaPose 模型转换

第二个模型是 AlphaPose 模型，onnx 模型可通过[地址](https://alphapose-model.obs.cn-north-4.myhuaweicloud.com/AlphaPose/fast_res50_256x192_bs1.onnx)获取。下载后使用模型转换工具 ATC 将 onnx 模型转换为 om 模型。

模型转换，步骤如下：

1. 将上述 onnx 模型下载至 `AlphaPose/models` 文件夹下，文件名为：`fast_res50_256x192_bs1.onnx` 。
2. 进入 `AlphaPose/models` 文件夹下执行命令：

```shell
atc --framework=5 --model=fast_res50_256x192_bs1.onnx --output=fast_res50_256x192_aipp_rgb --input_format=NCHW --input_shape="image:1,3,256,192" --soc_version=Ascend310 --insert_op_conf=aipp_192_256_rgb.cfg
```

执行该命令后会在当前文件夹下生成项目需要的模型文件 `fast_res50_256x192_aipp_rgb.om`。执行后终端输出为：

```shell
ATC run success, welcome to the next use.
```

表示命令执行成功。



## 5 准备

按照第 3 小结**软件依赖**安装 live555 和 ffmpeg，按照 [Live555离线视频转RTSP说明文档 ](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md)将 mp4 视频转换为 h264 格式。并将生成的 264 格式的视频上传到 `live/mediaServer` 目录下，然后修改 `AlphaPose/pipeline` 目录下的 `video.pipeline` 文件中 mxpi_rtspsrc0 的内容。

```
        "mxpi_rtspsrc0": {
            "props": {
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",      // 修改为自己开发板的地址和文件名
                "channelId": "0"
            },
            "factory": "mxpi_rtspsrc",
            "next": "mxpi_videodecoder0"
        },
```



## 6 编译与运行

**步骤1** 按照第2小结 **环境依赖** 中的步骤设置环境变量。

**步骤2** 按照第 4 小节 **模型转换** 中的步骤获得 om 模型文件，放置在 `AlphaPose/models` 目录下。

**步骤3** 按照第 5 小节 **准备** 中的步骤准备好输入视频流。

**步骤4** 编译。进入 `AlphaPose` 目录，在 `AlphaPose` 目录下执行命令：

```shell
bash build.sh
```

**步骤5** 运行视频流推理。在 `AlphaPose` 目录下执行命令：

```shell
bash run.sh video
```

命令执行成功后会在 `AlphaPose/out` 目录下生成 `alphapose.avi` 和 `alphapose.json` 文件，其中 `alphapose.avi` 为人体姿态可视化后的视频输出，`alphapose.json` 为每一帧图像中人物的关键点位置与置信度信息，查看文件验证人体关键点估计结果。

**步骤6** 运行图片推理。在 `AlphaPose` 目录下创建 `data` 目录，然后将需要推理的图片放在 `AlphaPose/data` 目录下。回到在 `AlphaPose` 目录下执行命令：

```shell
bash run.sh image
```

命令执行成功后会在 `AlphaPose/out` 目录下生成以测试图片名称命名的 json 文件，该文件包含图像中人物的关键点位置与置信度信息。查看文件验证人体关键点估计结果。



## 7 性能与精度测试

#### 7.1 性能测试

执行第 6 小节 **编译与运行** 中的步骤 1 至步骤 3 完成准备工作，进入 `AlphaPose` 目录，在 `AlphaPose` 目录下执行命令：

```shell
bash run.sh video --speedtest
```

结果将如下图所示，终端会每 10 帧打印一次当前帧数，前 10 帧的运行时间以及前10帧的平均帧率。

![性能测试结果](./image/speedtest.png)

[^注]: 输入视频帧率应高于25，否则无法发挥全部性能。且由于 Alphapose 人体关键点估计是一种自上而下的方式，所以实际推理速度与视频中的人数存在负相关关系，即人数越多，推理用时越多，速度越慢。上述展示的推理速度是在视频帧大小为 720*1280，且视频中只有一个人的条件下所得到的性能。

#### 7.2 精度测试

1.  安装 COCO 数据集 python API。

    ```shell
    pip3 install pycocotools
    ```

2.  下载 COCO VAL 2017 数据集，[验证集下载链接](http://images.cocodataset.org/zips/val2017.zip)，[验证集标签下载链接](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)。在`AlphaPose/src` 目录下创建 `dataset` 目录，将验证集和标签压缩文件解压至 `AlphaPose/src/dataset` 目录下。确保下载完数据集和标注文件后的 `AlphaPose/src` 目录结构为：

    ```
    .
    ├── dataset
    │   ├── annotations
    │   │   └── person_keypoints_val2017.json
    │   └── val2017
    │       ├── 000000581615.jpg
    │       ├── 000000581781.jpg
    │       └── other-images
    ├── evaluate.py
    ├── image.py
    └── video.py
    ```

    

3.  执行第 6 小节 **编译与运行** 中的步骤 1 至步骤 3 完成准备工作，进入 `AlphaPose` 目录，在 `AlphaPose` 目录下执行命令：

    ```shell
    bash run.sh evaluate
    ```

    命令执行结束后输出 COCO 格式的评测结果，并生成 val2017_keypoint_detect_result.json 检测结果文件。输出结果如下图所示：
    ![精度测试结果](./image/acctest.png)



## 8 常见问题

#### 8.1 输出视频无法播放

**问题描述：**

运行视频流推理时，发现生成的 `alphapose.avi` 无法播放。

**解决方案：**

检查 `AlphaPose/src/video.py` 中的 `VIDEO_WIDTH` 和 `VIDEO_HEIGHT` 参数，确保这两参数的值是输入的 .264 视频的宽和高。

