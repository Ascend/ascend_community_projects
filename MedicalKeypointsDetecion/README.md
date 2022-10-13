# 医学人体关键点检测技术

## 1 介绍
本项目主要用于研究辅助X线摄影的体位识别和体表解剖标志检测技术。

针对现有的关键点检测算法无法提供体位先验信息及无法进行体表解剖标志的精确定位的问题，本项目提出一种基于多任务学习的体位识别和体表解剖标志检测方法，该方法包含主干网络、体位分类分支、体表解剖标志检测分支和后处理模块，可以同时且高效的进行体位识别和体表解剖标志检测，是本领域的第一个方法。  

本开发样例基于MindX SDK实现医学人体关键点检测的功能，其主要流程为：1）利用FasterRcnn模型对输入图像进行人体检测，通过检测得到的box数据对图像进行仿射变换；2）利用HRNET-W48模型对仿射变换后的人像进行多任务学习，同时获取体位分类情况与体表标志检测；3）对上述结果进行后处理，生成一张带有医学人体检测点的人像图片。

样例输入：医学人体的jpg图片

样例输出：根据体位分类后的关键点标注图片

### 1.1 支持的产品

项目所用的硬件平台：Ascend310，Atlas 200DK板卡

### 1.2 支持的版本

支持的SDK版本为 npu-smi 20.2.0 

### 1.3 代码目录结构与说明

本工程名称为mmnet，工程目录如下图所示：

```
|-------- data                                // 存放测试图片
|-------- model                               // 存放模型
|-------- main.py                             // 主程序  
|-------- pipeline                               
|           |---- model1.pipeline              // 模型1pipeline配置文件
|           |---- model2.pipeline              // 模型2pipeline配置文件 
|-------- evaluate.py                         // 精度测试程序
|-------- README.md 
|-------- run.sh
```



## 2 环境依赖

| 软件名称  | 版本  |
| --------- | ----- |
| MindX SDK | 2.0.4 |
| python    | 3.7  |
| opencv2     | 4.6.0.66 |
| numpy       | 1.23.1   |
| pycocotools | 2.0.2    |
| pillow      |          |

注：安装pycocotools的命令为：

`conda install -c conda-forge pycocotools`


在编译运行项目前，需要设置环境变量：

- 环境变量介绍

```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

## 3 模型转换

### 3.1 FasterRcnn模型

该模型直接从华为仓内下载即可。

[Faster R-CNN-昇腾社区 (hiascend.com)](https://www.hiascend.com/zh/software/modelzoo/models/detail/C/8d8b656fe2404616a1f0f491410a224c)

并将该om文件放置在：“项目所在目录/model”

### 3.2 HRnet-w48模型

医学人体关键点检测采用自行训练的hrnet-w48模型。我们需要借助于ATC工具将onnx模型转化为对应的om模型。

具体步骤如下：
**步骤1** 获取模型onnx文件
，下载链接为[下载](https://mindx.sdk.obs.myhuaweicloud.com/ascend_community_projects/body_Keypoints_detection/pose_model.zip)

**步骤2** 将获取到的HRnet-w48模型onnx文件存放至：“项目所在目录/model”

**步骤3** 模型转换

在确保环境变量设置正确后，在onnx文件所在目录下执行以下命令：

```
atc --model=pose_model_384x288.onnx --framework=5 --output=pose_model_384_288_noAipp_fp16 --soc_version=Ascend310 --output_type=FP16
```

执行完模型转换脚本后，若提示如下信息说明模型转换成功，会在output参数指定的路径下生成pose_model_384_288_noAipp_fp16.om模型文件。

```python
ATC run success  
```

## 4 编译运行

接下来进行模型的安装运行，具体步骤如下：

**步骤1** 获取om模型（见前一章节）

**步骤2** 修改run.sh最后的执行文件名称”test.py"

**步骤3** 配置pipeline

根据所需场景，配置pipeline1与pipeline2文件，调整路径参数等。

model1.pipeline:

```python
"mxpi_tensorinfer0": {
			"props": {
				"modelPath": "./model/fasterrcnn_mindspore_dvpp.om"
			},
			"factory": "mxpi_tensorinfer",
			"next":"appsink0"
#修改om文件存放的路径
```

model2.pipeline:

```python
"mxpi_tensorinfer0": {
			"props": {
				"modelPath": "./model/pose_model_384_288_noAipp_fp16.om"
			},
			"factory": "mxpi_tensorinfer",
			"next":"appsink0"
#修改om文件存放的路径
```

**步骤4** 存放图片，执行模型进行测试

将测试图片存放至主目录下，修改main.py中的图片存放路径以及人像分割后的存储路径的相关代码：
[测试图片](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/body_Keypoints_detection/phto_for_Developers.zip)
【注】本模型定位场景为医学应用下的人体关键点检测，故只针对特定体态进行识别，建议从提供的测试图片或根据对应体态自行拍摄图片进行测试。

```
filepath = '' #测试图片路径
save_path = '' #测试结果保存路径
```

然后执行run.sh文件：

```
./run.sh
```

输出的图片即为样例的人像分割后的图片。

## 5 精度测试

对测试集中的3651张图片进行精度测试，具体步骤如下：

**步骤1** 获取测试集的图片,确保测试集的输入图片为jpg格式。获取测试集的标注文件（`person_keypoints_val2017.json`）。存放于data路径下。
（数据集暂未开源）

**步骤2** 修改evaluate.py中的测试集图片存放路径：

```
image_folder = ' ' #修改为测试集的路径
annotation_file = ' ' #修改为测试集的标注文件路径
txt = '' #体位分类情况
```

**步骤3** 修改run.sh最后的执行文件名称：

```
python3.9 evaluate.py
```

并执行：

```
./run.sh
```



