# AdaBins单目深度估计

## 1 介绍

基于AdaBins室内模型的单目深度估计，输出输入图像的深度图 \
输入：室内图片（图片格式为JPEG格式）\
输出：输入图片的深度图

### 1.1 支持的产品
Atlas 200DK

### 1.2 支持的版本
本项目配套的CANN版本为 [6.0.RC1](https://www.hiascend.com/software/cann/commercial) ，MindX SDK版本为 [3.0.RC2](https://www.hiascend.com/software/Mindx-sdk) 。

MindX SDK安装前准备可参考《用户指南》，[安装教程](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/quickStart/1-1安装SDK开发套件.md)

### 1.3 目录结构
```
|-------- dataset
|           |---- full_set                          // 测试集图片(需自行准备)
|           |---- depth_info                        // 测试集图片深度信息(需自行准备)
|           |---- test_set                          // 测试集图片子集，用于性能测试(可从full_set中复制少量图片)
|-------- models
|           |---- AdaBins_nyu.om                    // Adabins模型文件(需使用model_conversion.sh自行转换)
|           |---- aipp_adabins_640_480.aippconfig   // Adabins 模型转换配置文件
|           |---- model_conversion.sh               // 模型转换脚本
|-------- utils
|           |---- cal_accuracy.py                   // 计算模型精度脚本
|           |---- data_process.py                   // 处理数据集脚本
|-------- src
|           |---- CMakeLists.txt                    // 主程序CMakeLists文件
|           |---- main.cpp                          // 主程序
|           |---- Mono.cpp                          // 推理程序实现
|           |---- MOno.h                            // 推理程序头文件
|-------- results                                   // 用于存放推理结果的可视化图
|-------- build.sh                                  // 项目编译脚本
|-------- CMakeLists.txt                            // 项目CMakeLists文件
|-------- README.md                                 // ReadMe
|-------- test.jpg                                  // 运行所需图片(需自行准备)
```
## 2 环境依赖

### 2.1 软件版本
| 软件                 | 版本         | 说明                          | 获取方式                                                     |
| ------------------- | ------------ | ----------------------------- | ------------------------------------------------------------ |
| mxVision            | 3.0.RC2      | mxVision软件包                  | [链接](https://www.hiascend.com/software/Mindx-sdk) |
| Ascend-cann-toolkit | 6.0.RC1       | Ascend-cann-toolkit开发套件包    | [链接](https://www.hiascend.com/software/cann/commercial)    |
| Ubuntu            | 18.04 |                      |                                                |
| Python           | 3.9.12 |                      |                                               |

| Python软件                 | 版本         | 
| ------------------- | ------------ | 
| numpy            | 1.23.4   |
| imageio          | 2.21.0    | 
| h5py             | 3.7.0 | 
| opencv-python    | 4.6.0.66 | 
| Pillow           | 9.2.0 | 
| tqdm             | 4.63.0 | 

### 2.2 配置环境变量

在CANN的安装目录和MindX SDK的安装目录可以找到**set_env.sh**,将其加入bashrc文件中。

```bash
# 执行如下命令，打开.bashrc文件
cd $HOME
vi .bashrc

# 在.bashrc文件中添加以下语句
source ${CANN安装目录}/set_env.sh
source ${MindX SDK安装目录}/set_env.sh

# 保存退出.bashrc文件
# 执行如下命令使环境变量生效
source ~/.bashrc

# 查看环境变量
env
```

### 2.3 准备工作

> 模型转换

**步骤1** 下载[AdaBins_nyu.onnx](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/mindxsdk-referenceapps%20/contrib/MonocularDepthEstimation/AdaBins_nyu.onnx) 

**步骤2** 将下载得到的 `AdaBins_nyu.onnx` 放在 `model` 目录下

**步骤3** 修改模型转换脚本`model_conversion.sh`里的`install_path`

```bash
#此处请按实际情况修改
export install_path=/usr/local/Ascend/ascend-toolkit/latest 
```

**步骤4** 运行模型转换脚本 `model_conversion.sh` 

执行完模型转换脚本后，会生成相应的.om模型文件。

模型转换使用了ATC工具，如需更多信息请参考:

 https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/60RC1alpha003/infacldevg/atctool/atlasatc_16_0001.html

> 其他准备

**通用配置**
1) 在[main.cpp](./src/main.cpp) 中配置模型路径,设备ID
```c++
const uint32_t deviceID = 0;
const std::string modelPath = "models/AdaBins_nyu.om";
```

**可选配置**
1) 在[main.cpp](./src/main.cpp) 中配置数据集路径和保存路径以及模式等的默认值

   这些参数可以在运行程序时通过命令行参数传入，因此是可选修改项

```c++
std::string imagePath = "test.jpg";
std::string datasetPath = "dataset";
std::string outputPath = "results";
std::string mode = "run"; // `run`会保存经过可视化处理的深度图，`eval`会保存供精度计算的原始推理结果
```

2) 在[Mono.cpp](./src/Mono.cpp) 中配置输出可视化图大小
```c++
const cv::Size OUTPUT_SIZE(640,480);
```

**精度测试配置**

注意，由于原数据集较大，并且Atlas 200DK的内存有限，因此在Atlas 200DK上无法直接处理原数据集。需先在配置较高的服务器上进行数据预处理，然后复制到Atlas 200DK。

1) 下载测试集数据（ [下载地址](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) ）, 置于 [dataset](./dataset) 文件夹下，检查文件名是否为 `nyu_depth_v2_labeled.mat`

2) 运行 [utils/data_process.py](./utils/data_process.py)，处理完毕后的图片位于 [dataset/full_set](./dataset/full_set) 文件夹下，深度信息位于 [dataset/depth_info](./dataset/depth_info) 文件夹下 

   ```python
   python3 utils/data_process.py
   ```

3) 在[utils/cal_accuracy.py](./utils/cal_accuracy.py)中配置其他可选配置项
   ```python
   # thresholds for accuracy
   threshold_1 = 1.25
   threshold_2 = 1.25 ** 2
   threshold_3 = 1.25 ** 3
   ```

**性能测试配置**
1) 已经完成**精度测试配置**的可直接进行此处的第2步，否则请先按照**精度测试配置**进行前2步，

2) 将[dataset/full_set](./dataset/full_set)中7开头的文件拷贝至[dataset/test_set],共计111张图片。

   ```bash
   # 如果没有test_set文件夹，先创建文件夹
   mkidr dataset/test_set
   # 拷贝以7开头的共计111张图片，用作性能测试
   cp dataset/full_set/7* dataset/test_set
   ```

## 3 运行

请先准备自己的待测试图片，重命名为test.jpg，并放置于根目录下。

```bash
# 编译项目
bash build.sh

# 新建results文件夹，已经有该文件夹的应清空里面文件
mkdir results

# 运行
./main -i test.jpg -o results
```

执行完毕后，**经过可视化处理的深度图**会保存在工程目录下`results`中。

## 4 性能验证
```bash
# 进入工程目录
cd MonocularDepthEstimation_V2

# 编译项目，如果已经进行上一步则不需要重新编译
bash build.sh

# 新建results_perf文件夹，已经有该文件夹无需进行此步
mkdir results_perf

# 运行
./main -d dataset/test_set -m eval -o results_perf
```

执行完毕后，命令行会输出平均的每张图片处理时间。

作为对比，单目深度估计v1的111张图片的端到端总耗时为31.93s，平均每张图片的耗时为287.69ms。

## 5 精度验证
```bash
# 进入工程目录
cd MonocularDepthEstimation_V2

# 编译项目，如果已经进行上一步则不需要重新编译
bash build.sh

# 新建results_acc文件夹，已经有该文件夹的应清空里面文件
mkdir results_acc

# 运行
./main -d dataset/full_set -m eval -o results_acc
```

执行完毕后，**推理结果**会保存在工程目录下`results_acc`中。

因为数据集较大，以及需要计算的精度指标较多，如果使用的是Atlas 200DK开发者套件，因硬件配置限制，是无法在Atlas 200DK上完成计算这一步的。但是精度计算的结果仅与推理结果和标签相关，与环境无关，因此可以将整个项目同步至配置较高的服务器上完成计算，当然也可以仅迁移与精度计算相关的文件，如utils/cal_accuracy.py，dataset/depth_info文件夹和results_acc文件夹。服务器上需要安装numpy和opencv-python等Python包，可参照**2.1**准备环境。

```bash
# 在云服务器上，进入工程目录
cd MonocularDepthEstimation_V2

# 运行精度测试代码
python3 utils/cal_accuracy.py --gt_path=./dataset/depth_info --pred_path=./results_acc
```

执行完毕后，命令行会输出模型精度的各项指标。

作为对比，单目深度估计v1的平均精度为[0.7945507947291235, 0.9302372572859156, 0.9714240941130378]。