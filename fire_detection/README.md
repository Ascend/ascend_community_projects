# 火灾识别

## 1 介绍

### 1.1 概要描述

**火灾识别**项目是基于MindX SDK，在华为云昇腾平台上进行开发并部署到昇腾Atlas200 DK上，实现**对图像中是否为火灾场景进行识别**的功能。在Atlas200 DK上对火灾场景测试集进行推理，整体精度为94.15%，达到功能要求。

### 1.2 模型介绍

项目中用于火灾识别的是DenseNet模型，模型相关文件可以在此处下载：https://dl.dropbox.com/s/6t17srif65vzqfn/firedetect-densenet121-pretrained.pt

### 1.3 实现流程

1、在昇腾云服务器上搭建开发环境，设置环境变量配置mxVision、Ascend-CANN-toolkit。

2、下载模型：https://dl.dropbox.com/s/6t17srif65vzqfn/firedetect-densenet121-pretrained.pt ，将PyTorch模型转换为昇腾离线模型：densenet.pt  --> densemet.onnx --> densenet.om。

3、业务流程编排与配置。

4、python推理流程代码开发。

5、将项目移植至Atlas200 DK，测试精度性能。


火灾识别SDK流程图如图1-3所示：

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="pic/火灾检测SDK流程图.png">
    <br>
    <div style="color:orange; border-bottom: 1px #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1-3 火灾检测SDK流程图</div>
</center>



## 2 软件方案介绍

### 2.1 项目方案架构介绍

本项目架构主要流程为：图片进入流中，将图片解码之后放缩到特定尺寸，再利用densenet模型对图像中是否为火灾场景进行推理，最后输出结果。

表2.1 项目方案架构中各模块功能：

<center>

<div style="color:orange; border-bottom: 1px #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表2.1 火灾检测SDK流程图</div>

| 序号 | 子系统 | 功能描述 |
|:---:|:---:|:---:|
|1| 图像输入 | 调用Mindx SDK的appsrc输入图片 |
|2| 图像解码 | 调用Mindx SDK的mxpi_imagedecoder用于图像解码，当前只支持JPG/JPEG/BMP格式 |
|3| 图像放缩 | 调用Mindx SDK的mxpi_imageresize，放缩到224*224大小 |
|4| 火灾识别 | 调用Mindx SDK的mxpi_tensorinfer，选取torch框架下densenet模型对图像进行火灾预测 |
|5| 结果输出 | 调用Mindx SDK的appsink，提供端口从流中获取数据，用于输出结果 |


</center>

### 2.2 代码目录结构与说明

本工程名称为fire_detection,工程目录如下所示：

```
.
|-- README.md
|-- data
|   |-- other
|   `-- test
|-- envs
|   `-- atc_env.txt
|-- main.py
|-- models
|   `-- pth2onnx_310.py
|-- pipeline
|   `-- fire.pipeline
|-- run.sh
`-- test.py


```

## 3 设置开发环境


此项目在昇腾310云服务器开发并移植至Atlas200 DK上运行，需要在昇腾云服务器进行编译执行后在Atlas200 DK上搭**建运行环境**，并将项目部署到Atlas200 DK。


### 3.1 环境变量设置

查看文件 fire_detection/env/atc_env.txt 进行环境变量配置：

```bash

# 执行ascend-toolkit的set_env.sh
. /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh

# 导入SDK路径
export MX_SDK_HOME=${SDK安装路径}

export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"

export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0":"${MX_SDK_HOME}/lib/plugins"

export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib/modelpostprocessors":"${MX_SDK_HOME}/lib":"${MX_SDK_HOME}/opensource/lib":"${MX_SDK_HOME}/opensource/lib64":${LD_LIBRARY_PATH}

export PYTHONPATH=${MX_SDK_HOME}/python:$PYTHONPATH

```
### 3.2 densenet模型转换

**步骤1**   下载densenet模型，下载地址：https://dl.dropbox.com/s/6t17srif65vzqfn/firedetect-densenet121-pretrained.pt ，将*.pt模型放至 fire_detection/models/ 文件夹下。

**步骤2**   将*.pt模型转换为*.onnx模型,执行 fire_detection/models 下脚本 pth2onnx_310.py ，得到 densenet.onnx 模型。注意修改*.pt模型输入路径`MODEL_PATH = './models/firedetect-densenet121-pretrained.pt`，以及*.onnx模型输出路径`OUTPUT_MODEL_PATH = './models/densenet.onnx'`，将输出*.onnx模型保存在 ./models 文件夹下。

**步骤3**   执行`cd ./models`到 models 文件加下，执行atc命令 `atc --model=densenet.onnx --framework=5 --output=densenet  --insert_op_conf=fire.cfg --soc_version=Ascend310` ，利用atc工具将 densenet.onnx 转换为 densenet.om 。atc转换模型时使用aipp预处理，aipp配置内容 ./models/fire.cfg 如下：

```json

aipp_op {
    aipp_mode: static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : false
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 359
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 454
    matrix_r2c2 : 0
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128
    related_input_rank : 0
    mean_chn_0 : 0
    mean_chn_1 : 0
    mean_chn_2 : 0
    min_chn_0: 102.127
    min_chn_1: 94.401
    min_chn_2: 87.184
    var_reci_chn_0: 0.0137213737839432
    var_reci_chn_1: 0.0142654369859985
    var_reci_chn_2: 0.0143018549505871
}

```

### 3.3 进行图像推理

下载测试集data，将data下的test和other文件夹放至 fire_detection/data 下，在 fire_detection 路径下下运行 main.py 脚本，从 fire.pipeline 流中得到推理结果。

## 4 设置运行环境


关于Atlas200 DK硬件说明：https://support.huawei.com/enterprise/en/doc/EDOC1100223188/d8adcdca/product-introduction

整体流程为：先进行SD制卡，将Ubuntu烧入Atlas200 DK，烧录成功后通过ssh连接到Atlas200 DK安装CANN和Mindx SDK工具并设置环境变量，之后再根据需求安装相应依赖。

### 4.1 所需依赖准备

Atlas200 DK环境依赖软件和版本表4所示（建议按顺序安装）：

<center>

<div style="color:orange; border-bottom: 1px #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">表4.1 运行环境依赖</div>

| 依赖软件 | 版本  | 说明 |
| :--------: | :-----: | :------------------------: |
| Ubuntu | 18.04.1 LTS | 用于烧入Atlas200 DK |
| CANN | 6.0.RC1 | 用于提供模型转换ATC工具 |
| Mindx SDK | 3.0.RC2 | 用于提供pipeline插件等工具 |
| Python | 3.9.12 | 用于运行*.py文件 |
| numpy | 1.21.5 | 用于处理数组运算 |
| opencv-python | 4.6.0.66 | 用于图片处理 |



</center>

### 4.2 进行SD制卡

通过SD卡制作功能可以自行制作Atlas 200 DK开发者板的系统启动盘，完成Atlas 200 DK操作系统及驱动固件的安装。

一般制卡方式有两种，分别为直接mksd制卡和利用dd镜像制卡，后者更为快捷但无法充分地利用tf卡的空间，如果需要较大存储空间更推荐前者。

1.mksd制卡：https://www.hiascend.com/document/detail/zh/Atlas200DKDeveloperKit/1013/environment/atlased_04_0009.html

2.dd镜像制卡：https://bbs.huaweicloud.com/forum/thread-139685-1-1.html


制卡成功后将SD卡插至Atlas200 DK卡槽，插入电源启动开发板。

### 4.3 配置网络连接

进行网络配置前，需要确保Atlas 200 DK已完成操作系统及驱动的安装（通过制作SD卡的方式进行安装），且已正常启动。

通过ssh连接至Atlas200 DK后，配置Atlas200 DK连接网络：https://www.hiascend.com/document/detail/zh/Atlas200DKDeveloperKit/1013/environment/atlased_04_0012.html

进行配置后通过`ping www.huaweicloud.com`检查是否配置成功。

注意：如果进行网络共享时主机无法上网可能是Windows版本问题，可以通过版本回退的方法解决。

### 4.4 开发环境搭建

ssh连接Atlas200 DK成功后安装CANN和Mindx SDK，此时Atlas200 DK为davinci-mini环境，建议`su root`到root用户下进行安装和配置环境变量。

注意：CANN版本与Atlas200 DK环境匹配：https://gitee.com/ascend/tools/blob/master/makesd/Version_Mapping_CN.md

1.下载并安装CANN：https://www.hiascend.com/document/detail/zh/Atlas200DKDeveloperKit/1013/environment/atlased_04_0017.html

2.下载并安装Mindx SDK：

※ 安装CANN并且环境设置成功后再执行如下步骤
    
（1）下载与CANN版本匹配的 Mindx SDK：https://www.hiascend.com/zh/software/mindx-sdk/mxVision/community
    
（2）将下载的 *.run 移至Atlas200 DK：执行 `cd /usr/local/` ，在local文件夹下执行 `mkdir Mindx_SDK` 创建 Mindx_SDK 文件夹，将下载的 Ascend-mindxsdk_{version}_linux-aarch64.run 移至 /usr/local/Mindx_SDK 下。

（3）安装 mxVision ：执行 `cd /usr/local/Mindx_SDK` 后为 Ascend-mindxsdk_{version}_linux-aarch64.run 添加执行权限 ，成功后再执行 `Ascend-mindxsdk_{version}_linux-aarch64.run --install`安装 mxVision 。

（4）配置环境变量：安装成功后在 mxVision 路径下存在 set_env.sh ,用户在任意目录下执行` vi ~/.bashrc` 命令，打开 .bashrc 文件，在文件最后一行后面添加 `. /usr/local/Mindx_SDK/mxVision` ，执行 :wq! 命令保存文件并退出。执行 `source ~/.bashrc` 命令使其立即生效。

完成上述步骤后Atlas200 DK已成功安装CANN和Mindx SDK。


3.安装python、pip、opencv等依赖（Atlas200 DK为mini环境，建议通过miniconda安装其他库）：

（1）下载与Atlas200 DK Ubuntu版本对应的miniconda：https://docs.conda.io/en/latest/miniconda.html#linux-installers

（2）执行`exit`切换到HwHiAiUser用户下，为下载的 Miniconda3-latest-Linux-x86_64.sh 添加执行权限。

（3）HwHiAiUser用户下安装miniconda：`sh ./Miniconda3-latest-Linux-x86_64.sh`

（4）为root用户添加conda环境：`su root`后进入root用户，执行 `cat /home/HwHiAiUser/.bashrc` ，查看HwHiAiUser用户的 .bashrc , 复制 >>> conda initialize >>> 至 <<< conda initialize <<< 的内容，将复制内容粘贴至 ~/.bashrc ，执行 `source ~/.bashrc` 为root添加conda环境。

（5）将Atlas200 DK重启reset，再次进入root用户即可看到conda环境。

（6）利用conda在root用户下用`conda install`安装numpy、opencv-python等库。

Atlas200 DK已经成功安装相关依赖。


## 5 部署与运行

**步骤1** 部署及环境设置：

将 fire_detection 文件夹移植到Atlas200 DK的用户HwHiAiUser下，执行命令 `su root` 进入root用户下(默认密码：Mind@123)后，执行 `cd /home/HwHiAiUser/fire_detection` 进入fire_detection路径下。

fire_detection应包含：
```
.
|-- README.md
|-- data
|   |-- other
|   `-- test
|-- envs
|   `-- atc_env.txt
|-- main.py
|-- models
|   |-- densenet121-pretrained.pt
|   |-- densenet.onnx
|   |-- densenet.om
|   `-- pth2onnx_310.py
|-- pipeline
|   `-- fire.pipeline
|-- run.sh
`-- test.py


```

**步骤2** 推理运行：

在执行步骤一命令后的路径下运行脚本 main.py 在开发板上进行推理。

**步骤3** 输出结果：

```
精度： 94.15041782729804 %
总耗时： 3534.153747000346 ms   总图片数： 359
平均单张耗时： 9.8444394066862 ms
```
精度为94.15%，与原模型推理精度误差小于1%;平均图像单张耗时9.84ms,满足25 fps要求。

## 6 常见问题

### 6.1 无法调用plugin插件

**问题描述：**

在Atlas200 DK上运行 mian.py 无法调用plugin插件。

**解决方案：**

确保root环境下可以执行atc，并且能够调用mindx后，在root用户下执行 main.py 脚本文件。

### 6.2 推理结果精度不达标

**问题描述：**

在开发环境的推理结果精度不达标。

**解决方案：**

在利用atc工具进行模型转换时配置aipp，提升模型精度：

```json

aipp_op {
    aipp_mode: static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : false
    matrix_r0c0 : 256
    matrix_r0c1 : 0
    matrix_r0c2 : 359
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 454
    matrix_r2c2 : 0
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128
    related_input_rank : 0
    mean_chn_0 : 0
    mean_chn_1 : 0
    mean_chn_2 : 0
    min_chn_0: 102.127
    min_chn_1: 94.401
    min_chn_2: 87.184
    var_reci_chn_0: 0.0137213737839432
    var_reci_chn_1: 0.0142654369859985
    var_reci_chn_2: 0.0143018549505871
}

```
