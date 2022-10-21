# DeepSORT目标跟踪

## 1 介绍

 DeepSOR目标跟踪后处理插件基于MindXSDK开发，在晟腾芯片上进行目标检测和跟踪，可以对行人进行画框和编号，将检测结果可视化并保存。项目主要流程为：通过live555服务器进行拉流输入视频，然后进行视频解码将264格式的视频解码为YUV格式的图片，图片缩放后经过模型推理进行行人识别，识别结果经过FairMOT后处理后得到识别框，对识别框进行跟踪并编号，用编号覆盖原有的类别信息，再将识别框和类别信息分别转绘到图片上，最后将图片编码成视频进行输出。 

### 1.1 支持的产品

昇腾310(推理)

### 1.2 支持的版本

本样例配套的CANN版本为[5.0.4](https://www.hiascend.com/software/cann/commercial)。支持的SDK版本为[2.0.4](https://www.hiascend.com/software/Mindx-sdk)。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 软件方案介绍

基于MindX SDK的 DeepSORT目标识别业务流程为：待检测视频存放在live555服务器上经mxpi_rtspsrc拉流插件输入，然后使用视频解码插件mxpi_videodecoder将视频解码成图片，再通过图像缩放插件mxpi_imageresize将图像缩放至满足检测模型要求的输入图像大小要求，缩放后的图像输入模型推理插件mxpi_tensorinfer得到检测结果，本项目开发的DeepSORT后处理插件处理推理结果，得到识别框。再接入跟踪插件中识别框进行目标跟踪，得到目标的跟踪编号，然后在使用本项目开发的mxpi_trackidreplaceclassname插件将跟踪编号覆盖类名信息，使用mxpi_object2osdinstances和mxpi_opencvosd分别将识别框和类名（存储跟踪编号）绘制到原图片，再通过mxpi_videoencoder将图片合成视频。

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统               | 功能描述                                                     |
| ---- | -------------------- | :----------------------------------------------------------- |
| 1    | 视频输入             | 接收外部调用接口的输入视频路径，对视频进行拉流，并将拉取的裸流存储到缓冲区（buffer）中，并发送到下游插件。 |
| 2    | 视频解码             | 用于视频解码，当前只支持H264/H265格式。                      |
| 3    | 数据分发             | 对单个输入数据分发多次。                                     |
| 4    | 数据缓存             | 输出时为后续处理过程另创建一个线程，用于将输入数据与输出数据解耦，并创建缓存队列，存储尚未输出到下流插件的数据。 |
| 5    | 图像处理             | 对解码后的YUV格式的图像进行指定宽高的缩放，暂时只支持YUV格式 的图像。 |
| 6    | 模型推理插件         | 目标分类或检测，目前只支持单tensor输入（图像数据）的推理模型。 |
| 7    | 模型后处理插件       | 实现对DeepSORT模型输出的tensor解析，获取目标检测框以及对应的ReID向量，传输到跟踪模块。 |
| 8    | 跟踪插件             | 实现多目标（包括机非人、人脸）路径记录功能。                 |
| 9    | 跟踪编号取代类名插件 | 用跟踪插件产生的编号信息取代后处理插件产生的类名信息，再将数据传入数据流中。 |
| 10   | 目标框转绘插件       | 将流中传进的MxpiObjectList数据类型转换可用于OSD插件绘图所使用的的 MxpiOsdInstancesList数据类型。 |
| 11   | OSD可视化插件        | 主要实现对每帧图像标注跟踪结果。                             |
| 12   | 视频编码插件         | 用于将OSD可视化插件输出的图片进行视频编码，输出视频。        |

### 1.4 代码目录结构与说明

本工程名称为 DeepSORT，工程目录如下图所示：

```
├── models
│   └── aipp_FairMOT.config            # 模型转换aipp配置文件
├── pipeline
│   └── deepsort.pipeline        # pipeline文件
├── plugins
│   ├── FairmotPostProcess     #DeepSORT后处理插件
│       └── move_so.sh
│   ├── Deepsort     # DeepSORT的Tracking
│   │   ├── CMakeLists.txt        
│   │   ├── DeepSort.cpp  
│   │   ├── DeepSort.h
│   │   ├── move_so.sh
│   │   └── build.sh
│   ├── Deepsort .patch  
│   ├── DeepAppearanceDescriptor.patch
│   ├── MunkresAssignment.patch  
│   └── MxpiTrackIdReplaceClassName  # 跟踪编号取代类名插件
│       └── move_so.sh
├── CMakeLists.txt
├── build.sh
├── main.cpp
└── run.sh
```


## 2 环境依赖

推荐系统为ubantu 18.04，环境依赖软件和版本如下表：

| 软件名称            | 版本        | 说明                          | 获取方式                                                     |
| ------------------- | ----------- | ----------------------------- | ------------------------------------------------------------ |
| MindX SDK           | 2.0.4       | mxVision软件包                | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| ubantu              | 18.04.1 LTS | 操作系统                      | Ubuntu官网获取                                               |
| Ascend-CANN-toolkit | 5.0.4       | Ascend-cann-toolkit开发套件包 | [链接](https://www.hiascend.com/software/cann/commercial)    |

在编译运行项目前，需要设置环境变量：

```
export MX_SDK_HOME=${MX_SDK_HOME}
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_SCANNER=${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export GST_PLUGIN_PATH=${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins
```

注：其中SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径;install_path替换为开发套件包所在路径。LD_LIBRARY_PATH用以加载开发套件包中lib库。



## 3 软件依赖

推理中涉及到第三方软件依赖如下表所示。

| 依赖软件 | 版本       | 说明                           | 使用教程                                                     |
| -------- | ---------- | ------------------------------ | ------------------------------------------------------------ |
| live555  | 1.09       | 实现视频转rstp进行推流         | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) |
| ffmpeg   | 2021-07-21 | 实现mp4格式视频转为264格式视频 | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/pc%E7%AB%AFffmpeg%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B.md#https://ffmpeg.org/download.html) |



## 4 模型转换

本项目中适用的模型是FairMOT模型，onnx模型可以直接[下载](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/FairMOT/mot_v2.onnx)。下载后使用模型转换工具 ATC 将 onnx 模型转换为 om 模型，模型转换工具相关介绍参考链接：https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html 。

模型转换，步骤如下：

1. 从上述 onnx 模型下载链接中下载 onnx 模型至 `deepsort/models` 文件夹下，文件名为：mot_v2.onnx 。
2. 进入 `deepsort/models` 文件夹下执行命令：

```
atc --input_shape="input.1:1,3,480,864" --check_report=./network_analysis.report --input_format=NCHW --output=./mot_v2 --soc_version=Ascend310 --insert_op_conf=./aipp_FairMOT.config --framework=5 --model=./mot_v2.onnx
```

执行该命令后会在当前文件夹下生成项目需要的模型文件 mot_v2.om。执行后终端输出为：

```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

表示命令执行成功。



## 5 准备

按照第3小结**软件依赖**安装live555和ffmpeg，按照 [Live555离线视频转RTSP说明文档](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/Live555%E7%A6%BB%E7%BA%BF%E8%A7%86%E9%A2%91%E8%BD%ACRTSP%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md)将mp4视频转换为h264格式。并将生成的264格式的视频上传到`live/mediaServer`目录下，然后修改`FairMOT/pipeline`目录下的fairmot.pipeline文件中mxpi_rtspsrc0的内容。

```
        "mxpi_rtspsrc0": {
            "factory": "mxpi_rtspsrc",
            "props": {
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",      // 修改为自己所使用的的服务器和文件名
                "channelId": "0"
            },
            "next": "mxpi_videodecoder0"
        },
```



## 6 编译与运行

**步骤1** 按照第2小结**环境依赖**中的步骤设置环境变量。按照第5小结**准备**完成相关安装和修改。

**步骤2** 按照第 4 小节 **模型转换** 中的步骤获得 om 模型文件，放置在 `deepsort/models` 目录下。

**步骤3** 进入plugins目录，将FairmotPostProcess和MxpiTrackIdReplaceClassName插件以如下结构放在 `plugins` 目录下。[链接]https://gitee.com/ascend/mindxsdk-referenceapps/tree/master/contrib/FairMOT。
将DeepSort开源代码的DeepSort、DeepAppearanceDescriptor和MunkresAssignment以如下结构放在 `plugins` 目录下。[链接]https://github.com/shaoshengsong/DeepSORT 。
├── plugins
│   ├── FairmotPostProcess     #DeepSORT后处理插件
│   │   ├── CMakeLists.txt        
│   │   ├── FairmotPostProcess.cpp  
│   │   ├── FairmotPostProcess.h
│   │   ├── move_so.sh
│   │   └── build.sh
│   └── MxpiTrackIdReplaceClassName  # 跟踪编号取代类名插件
│       ├── CMakeLists.txt
│       ├── MxpiTrackIdReplaceClassName.cpp
│       ├── MxpiTrackIdReplaceClassName.h
│       ├── move_so.sh
│       └── build.sh
│   ├── DeepSort     # DeepSORT的Tracking
│   │   ├── CMakeLists.txt        
│   │   ├── DeepSort.cpp  
│   │   ├── DeepSort.h
│   │   ├── kalmanfilter.cpp 
│   │   ├── kalmanfilter.h
│   │   ├── linear_assignment.cpp
│   │   ├── linear_assignment.h
│   │   ├── nn_matching.cpp
│   │   ├── nn_matching.h
│   │   ├── tracker.cpp 
│   │   ├── tracker.h
│   │   ├── track.cpp
│   │   ├── track.h
│   │   ├── move_so.sh
│   │   └── build.sh
│   ├── DeepAppearanceDescriptor    #特征
│   │   ├── FeatureTensor.cpp
│   │   ├── FeatureTensor.h
│   │   ├── model.cpp
│   │   ├── model.h
│   │   └── dataType.h
│   ├── MunkresAssignment
│   │   ├── hungarianoper.cpp
│   │   ├── hungarianoper.h
│   │   └── munkres
│   │        ├── munkres.cpp
│   │        ├── munkres.h
│   │        └── matrix.h
在 `plugins` 目录下执行命令：
```
patch -p1 < MxpiTrackIdReplaceClassName.patch 
patch -p1 < FairmotPostProcess.patch 
patch -p1 < MunkresAssignment.patch  
patch -p1 < DeepAppearanceDescriptor.patch  
patch -p1 < DeepSort.patch  
```
注：如果Fairmot文件有更新，则需要回退历史版本。
```
git clone https://gitee.com/ascend/mindxsdk-referenceapps.git
cd /contrib/FairMOT
git reset c2e5c1a51eff4214563d3993a742183e1ff9e55c --hard
```
**步骤4** 编译。进入 `deepsort` 目录，在 `deepsort` 目录下执行命令：

```
bash build.sh
```
注：执行bash步骤前，需要将plugins中所有插件里CMakeLists.txt、move_so.sh文件和主目录下的CMakeLists.txt文件的SDK安装路径${MX_SDK_HOME}替换为用户的SDK安装路径

**步骤5** 运行。回到deepsort目录下，在deepsort目录下执行命令：

```
bash run.sh
```

命令执行成功后会在当前目录下生成检测结果视频文件out.h264,查看文件验证目标跟踪结果。

## 7 性能测试

**测试帧率：**

按照第6小结编译与运行中的步骤进行编译运行，服务器会输出运行到该帧的平均帧率。


注：输入视频帧率应高于25，否则无法发挥全部性能。

## 8 精度测试

**测试性能：**
以MOT16 数据集为基准测试模型的 MOTA 值。 
执行步骤：
**步骤1** 下载 MOT16数据集，下载链接https://motchallenge.net/。在deepsort 目录下创建 /py-motmetrics-develop/ 目录，下载py-motmetrics工具，下载链接https://github.com/cheind/py-motmetrics。
将MOT16 数据集的gt文件放入/py-motmetrics-develop/motmetrics/data/train/gt/1/gt/。将自己运行得到的txt文件放入/py-motmetrics-develop/motmetrics/data/train/，并将命名改为1.txt.
注意：自己运行得到的txt文件是按照第一列（frame id）的数值进行排序的，需要先将其按照第二列（track id）的数值进行排序后放入。

**步骤2** 安装 pycocotools 评测工具包。执行命令：
```
pip install motmetrics
```

**步骤3** 在/py-motmetrics-develop/motmetrics/apps/目录下运行命令
```
python eval_motchallenge.py .../deepsort/py-motmetrics-develop/motmetrics/data/train/gt/ ..../deepsort/py-motmetrics-develop/motmetrics/data/train

```
注：每次运行得到的txt文件都需要删除并重新生成。
      因为视频流是循环输入，故在此测试精度时候选取视频前九百帧进行测试，如测试需要，可修改deepsort/plugins/DeepSort/DeepSort.cpp中的全局变量control值，进行精度测试。

## 9 常见问题

9.1 测试输入视频流不存在或断流

**问题描述：**

运行时报错：
执行失败，提示 ：“connect stream failed.”


**解决方案：**

确保使用 live555 工具进行推流：在deepsort 目录下下载live555并解压；将测试用例放在deepsort/live/mediaServer/ 目录下；回到 deepsort/ 文件夹，在此文件夹下运行命令： bash run.sh

