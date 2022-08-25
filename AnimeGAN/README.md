# AnimeGAN图像风格转换

## 1 介绍

### 1.1 概要描述

本样例基于MindxSDK开发，可以实现**从真实世界图片到动漫风格图片的风格转换**，在昇腾芯片上对输入图片进行图像**风格转换**并将转换结果保存，支持多档次的动态分辨率输入。

论文原文：https://link.springer.com/chapter/10.1007/978-981-15-5577-0_18

Github仓库：https://github.com/TachibanaYoshino/AnimeGAN

预训练模型和数据集：https://github.com/TachibanaYoshino/AnimeGAN/releases 
或者obs://animegan-mxsdk/models以及obs://animegan-mxsdk/dataset

### 1.2 软件方案介绍

基于MindX SDK的AnimeGAN风格转换模型的推理流程为：

首先使用openCV将待转换图片缩放至合适的分辨率档位，然后通过appsrc插件输入，然后使用图像解码插件mxpi_imagedecoder对图片进行解码，再通过图像标准化插件mxpi_imagenormalize将图片数据标准化到[-1,1],然后输入模型推理插件mxpi_tensorinfer，最后通过自定义的后处理插件anmieganpostprocess得到对推理结果进行后处理并保存。本系统的各模块及功能如表1所示：

表1 系统方案各模块功能描述：

| 序号 | 子系统     | 功能描述                                                               |
| ---- | ---------- | ---------------------------------------------------------------------- |
| 1    | 图片输入   | 调用MindX SDK的appsrc输入图片                                          |
| 2    | 图片解码   | 调用MindX SDK的mxpi_imagedecoder解码图片为RGB数据                      |
| 3    | 图片标准化 | 调用MindX SDK的mxpi_imagenormalize将图片数据标准化到[-1,1]             |
| 4    | 模型推理   | 调用MindX SDK的mxpi_tensorinfer对输入张量进行推理                      |
| 5    | 后处理     | 调用自定义的后处理插件对模型推理输出映射回[0,255]，转换成RBG数据并保存 |

### 1.4 代码目录结构与说明

项目名称为AnimeGAN，项目目录如下图所示：

```
│  README.md
│  main.py                  # 主程序
|  eval.py                  # 测量NPU生成的图片与Tensorflow框架生成的图片的SSIM
|  animegan.pipeline        # pipeline文件
├─dataset
├─results
├─models
│      AnimeGAN_FD.om       # 应用推理所需的om模型
├─plugins
│  ├─AnimeGANPostProcessor  # AnimeGAN后处理插件
│  │      build.sh          #编译AnimeGAN后处理插件的脚本
│  │      CMakeLists.txt
│  │      AnimeGANPostProcessor.cpp
│  │      AnimeGANPostProcessor.h
```

## 2 环境依赖

推荐系统为ubuntu 18.04，环境依赖软件和版本如下表：

| 软件名称 | 版本    |
| -------- | ------- |
| cmake    | 3.5.+   |
| mxVision | 3.0.RC2 |
| Python   | 3.9.12  |
| CANN     | 5.1.RC1 |
| gcc      | 7.5.0   |

python第三方库依赖如下表：

| 软件名称      | 版本     | 说明                                                                 |
| ------------- | -------- | -------------------------------------------------------------------- |
| opencv-python | 4.6.0.66 |
| scikit-image  | 0.19.3   | 仅在eval.py中依赖，用于测量Tensorflow框架生成图片和NPU生成图片的SSIM |

模型转换所需ATC工具环境搭建参考链接：[参考链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1alpha02/infacldevg/atctool/atlasatc_16_0004.html)。

在编译运行项目前，需要设置环境变量：

```bash
export MX_SDK_HOME=${SDK安装路径}/mxVision
export LD_LIBRARY_PATH="${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${MX_SDK_HOME}/python:${PYTHONPATH}"
export GST_PLUGIN_SCANNER="${MX_SDK_HOME}/opensource/libexec/gstreamer-1.0/gst-plugin-scanner"
export GST_PLUGIN_PATH="${MX_SDK_HOME}/opensource/lib/gstreamer-1.0:${MX_SDK_HOME}/lib/plugins"
```

亦可在CANN以及MindX SDK的安装目录找到set_env.sh,并运行脚本(**推荐此种方式**)：

```bash
bash ${SDK安装路径}/set_env.sh
bash ${CANN安装路径}/../set_env.sh
```

## 3 模型转换

本项目推理模型权重采用官方发布的预训练模型：[Tensorflow模型下载链接](https://github.com/TachibanaYoshino/AnimeGAN/releases)。

使用之前须将Tensorflow的ckpt模型文件转换成pb文件，再使用转化工具ATC将模型转化为om模型。ckpt模型文件转换成pb文件的相关介绍参考[此处](https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/contrib/cv/BicycleGAN_ID1287_for_TensorFlow/bicyclegan_pb_frozen.py)。已经转换好的pb模型和支持动态分辨率的om模型可通过华为OBS工具在obs://animegan-mxsdk/models/获得。

模型转换工具相关介绍参考[CANN文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1alpha02/infacldevg/atctool/atlasatc_16_0005.html)。

首先进行ATC工具参考环境变量配置，若采用**2 环境依赖**一节所列的set_env.sh方式，此步可以略过：

```bash
# 设置环境变量（请确认install_path路径是否正确）
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

下载或转换成pb模型之后，将pb文件移至models目录下,并在终端执行如下命令：

```bash
atc  --output_type="generator/G_MODEL/output:0:FP32" --input_shape="test:1,-1,-1,3" --out_nodes="generator/G_MODEL/output:0" --input_format=NHWC --output="models/AnimeGAN_FD" --soc_version=Ascend310 --dynamic_image_size="384,384;384,512;384,640;384,768;384,896;384,1024;384,1152;384,1280;384,1408;384,1536;512,384;512,512;512,640;512,768;512,896;512,1024;512,1152;512,1280;512,1408;512,1536;640,384;640,512;640,640;640,768;640,896;640,1024;640,1152;640,1280;640,1408;640,1536;768,384;768,512;768,640;768,768;768,896;768,1024;768,1152;768,1280;768,1408;768,1536;896,384;896,512;896,640;896,768;896,896;896,1024;896,1152;896,1280;896,1408;896,1536;1024,384;1024,512;1024,640;1024,768;1024,896;1024,1024;1024,1152;1024,1280;1024,1408;1024,1536;1152,384;1152,512;1152,640;1152,768;1152,896;1152,1024;1152,1152;1152,1280;1152,1408;1152,1536;1280,384;1280,512;1280,640;1280,768;1280,896;1280,1024;1280,1152;1280,1280;1280,1408;1280,1536;1408,384;1408,512;1408,640;1408,768;1408,896;1408,1024;1408,1152;1408,1280;1408,1408;1408,1536;1536,384;1536,512;1536,640;1536,768;1536,896;1536,1024;1536,1152;1536,1280;1536,1408;1536,1536" --framework=3 --model="models/AnimeGAN.pb" --precision_mode=force_fp32
```

> 以上命令将模型转化为具有多档位的动态分辨率模型，转成单档位固定分辨率可使用如下命令：

```bash
atc  --output_type="generator/G_MODEL/output:0:FP32" --input_shape="test:1,864,864,3" --out_nodes="generator/G_MODEL/output:0"  --input_format=NHWC --output="models/AnimeGAN_864" --soc_version=Ascend310 --framework=3 --model="models/AnimeGAN.pb" --precision_mode=force_fp32
```

> 如使用固定分辨率命令转化后的模型，在使用mxpi_imageresizer缩放插件下游紧接模型推理时，可以自动获取缩放宽高，在pipeline中mxpi_imageresizer插件无需再配置resizeHeight和resizerWidth属性。
>
> 而使用多档位命令转换的模型时，未配置宽高属性则会默认缩放到第一个档位，使用其它档位仍需配置缩放的宽高。 亦可不使用mxpi_imageresizer插件，而是使用预处理将图片缩放至模型包含的档位。

## 4 AnimeGAN风格转换推理流程开发实现

### 4.1 pipeline编排

```
    appsrc						# 输入
    mxpi_imagedecoder			# 图像解码
    mxpi_tensorinfer			# 模型推理（风格转换）
    animeganpostprocessor	# 模型后处理
    appsink						# 输出
```

### 4.2 AnimeGAN后处理库开发

一般的后处理插件开发，可以根据任务类型，选择SDK已经支持的后处理基类去派生一个新的子类，这些后处理基类分别为目标检测，分类任务，语义分割，文本生成。但因为SDK现有支持的后处理插件的基类并不包含GAN，因此需要新增后处理基类继承PostProcessBase，并写入新的数据结构。

> 参考链接：https://www.hiascend.com/document/detail/zh/mind-sdk/30rc2/vision/mxvisionug/mxvisionug_0057.html

或者基于普通插件的开发步骤，自行实现后处理的功能。

> 参考链接：https://www.hiascend.com/document/detail/zh/mind-sdk/30rc2/vision/mxvisionug/mxvisionug_0048.html

### 4.3 主程序开发

1、初始化流管理。 

2、加载图像，对图像进行预处理以符合动态分辨率模型的档位

3、向流发送图像数据，进行推理。

4、获取pipeline各插件输出结果，其中animeganpostprocessor会自动保存结果。

5、销毁流

## 5 编译与运行

- 编译

  ```bash
  bash plugins/AnimeGANPostProcessor/build.sh
  # 运行脚本后，会在plugins/AnimeGANPostProcessor/lib/plugins下生成so文件
  cp plugins/AnimeGANPostProcessor/lib/plugins/libanimeganpostprocessor.so ${MX_SDK_HOME}/lib/plugins
  # 将该文件拷贝至MindX SDK的plugins目录中,注意文件权限应为640
  ```
  
- 运行

  ``` python
  #更改main.py中的DATA_PATH为测试图片所在的文件夹，并运行命令
  DATA_PATH = "dataset/test/HR_photo"
  ```
  ```bash
  python main.py
  ```

> 1.先获取pb模型，并使用命令转化成所需要的om模型，并放置在models目录下。
>
> 2.所需的环境变量推荐使用CANN和MindX SDK安装目录下的set_env.sh进行导入。
>
> 3.运行时请准备图片，按上述命令执行后，风格转换结果会生成在results/npu路径下,该路径可以在pipeline中修改animeganpostprocessor的outputPath属性来变更。


## 6 常见问题

### 6.1 后处理库权限问题

**问题描述：**

提示Check Owner permission failed: Current permission is 7, but required no greater than 6.

**解决方案：**

后处理库so文件权限太高，需要降低权限至640,参见**5 编译与运行**一节有关编译的内容。