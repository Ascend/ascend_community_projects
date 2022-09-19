# ChineseOCR

## 1 介绍
本开发样例演示中文字体识别ChineseOCR，供用户参考。 本系统基于昇腾Atlas300卡。主要为单行中文识别系统，系统将图像进行适当的仿射变化，然后送入字符识别系统中进行识别后将识别结果输出。

### 1.1 支持的产品

本系统采用Atlas300-3010作为实验验证的硬件平台，并支持Atlas200RC以及Atlas500的硬件平台.具体产品实物图和硬件参数请参见《Atlas 300 AI加速卡 用户指南（型号 3010）》。由于采用的硬件平台为含有Atlas 300的Atlas 800 AI服务器 （型号3010），而服务器一般需要通过网络访问，因此需要通过笔记本或PC等客户端访问服务器，而且展示界面一般在客户端。

### 1.2 支持的版本

支持1.75.T11.0.B116, 1.75.T15.0.B150, 20.1.0

版本号查询方法，在Atlas产品环境下，运行以下命令：

```
npu-smi info
```



### 1.3 软件方案介绍

软件方案将主要为中文字符识别系统。系统功能具体描述请参考 表1.1 系统方案各功能描述。系统可以实现将字符检测子系统的结果中的文字进行识别。本方案选择使用crnn作为字符识别模型。系统方案中各模块功能如表1.2 所示。

表1.1 系统方案各子系统功能描述：

| 序号 | 子系统   | 功能描述                                                     |
| ---- | -------- | ------------------------------------------------------------ |
| 1    | 字符识别 | 从pipeline中读取到输入的图片，然后将图片放缩为固定大小，放缩的大小与模型的输入有关，然后将放缩后的结果送入字符识别系统，放缩的大小与模型的输入大小有关，之后将结果送入到文字识别模型进行文字识别，并将识别结果进行输出。 |

表1.2 系统方案中各模块功能：

| 序号 | 子模块     | 功能描述                                                     |
| ---- | ---------- | ------------------------------------------------------------ |
| 1    | 输入图像   | 将图像（JPG格式）通过本地代码输入到pipeline中。              |
| 2    | 图像解码   | 通过硬件（DVPP）对图像进行解码，转换为UINT8格式进行后续处理。 |
| 3    | 图形放缩   | 由于文本检测模型的输入大小为固定的维度，需要使用图像放缩插件将图像等比例放缩为固定尺寸。 |
| 4    | 图像归一化 | 将放缩之后的图像送入归一化插件中得到归一化结果。             |
| 5    | 文字识别   | 在图像放缩后，将缓存区数据送入文字识别模型。本方案选用crnn进行文本识别。 |



### 1.4 代码目录结构与说明

本Sample工程名称为ChineseOCR，工程目录如下图1.2所示：

```
├── IDCardRecognition
│   ├── ChineseOCR.py
│   └── README.md
├── pipeline
│   ├── chineseocr.pipeline
├── models
│   ├── crnn
├── data
│   ├── inputdata
```

### 1.5 技术实现流程图

![pic](./RESOURCES/flow.png)



### 1.6 特性及适用场景

任意长宽高度的的单行中文字符图片

## 2 环境依赖

推荐系统为ubuntu 18.04或centos 7.6，环境依赖软件和版本如下表：

| 软件名称 | 版本   |
| -------- | ------ |
| Python   | 3.9.12 |
| protobuf | 3.19.0 |
| google   | 3.0.0  |

在编译运行项目前，需要设置环境变量：

- `ASCEND_HOME` Ascend安装的路径，一般为 `/usr/local/Ascend`
- `DRIVER_HOME` 可选，driver安装路径，默认和$ASCEND_HOME一致，不一致时请设置
- `ASCEND_VERSION` acllib 的版本号，用于区分不同的版本，参考$ASCEND_HOME下两级目录，一般为 `ascend-toolkit/*version*`
- `ARCH_PATTERN` acllib 适用的 CPU 架构，查看`$ASCEND_HOME/$ASCEND_VERSION`文件夹，可取值为 `x86_64-linux` 或 `arm64-linux`等

```
export ASCEND_HOME=/usr/local/Ascend
export DRIVER_HOME=/usr/local/Ascend
export ASCEND_VERSION=ascend-toolkit/latest
export ARCH_PATTERN=x86_64-linux
```



## 依赖安装

使用pip安装所需的插件



## 运行

**步骤1** 将`crnn`模型放到`models/paddlecrnn`文件夹内

**步骤2**配置环境变量，根据自己的环境变量不同，需要配置不同的环境变量，下面给出参考示例：

```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_AICPU_PATH=${XXX}/Ascend/ascend-toolkit/latest
export ASCEND_OPP_PATH=${XXX}/Ascend/ascend-toolkit/latest/opp
export ASCEND_HOME_PATH=${XXX}/Ascend/ascend-toolkit/latest
export GST_PLUGIN_SCANNER=/home/liuyipeng2/NewSDK/mxVision-3.0.RC2/opensource/libexec/gstreamer-1.0/gst-plugin-scanner
export MX_SDK_HOME=/home/liuyipeng2/NewSDK/mxVision-3.0.RC2
```

**步骤3** 在chineseocr.py`中，更改`pipeline路径

**步骤4** 运行chineseocr.py文件得到中文识别结果





## 5 软件依赖说明



| 依赖软件 | 版本   | 说明                                         |
| -------- | ------ | -------------------------------------------- |
| glob     | 0.7    | 数据查找，并将搜索的到的结果返回到一个列表中 |
| protobuf | 3.19.0 | 数据序列化反序列化组件                       |



## 6 常见问题

### 6.1 输入图片大小与模型不匹配问题

**问题描述：**

运行失败：

```
E20220826 10:05:45.466817 19546 MxpiTensorInfer.cpp:750] [crnn recognition][1001][General Failed] The shape of concat inputTensors[0] does not match model inputTensors[0]
...
```

**解决方案：**

在imagedecode插件，设定解码方式的参数为opencv，选择模型格式为RGB，然后再imageresize插件里面设定o解码方式为opencv