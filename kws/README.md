# Keyword Spotting KWS

## 1 简介
  本开发样例基于MindX SDK实现了端到端的关键词检测（Keyword Spotting KWS）。<br/>
  所选择的关键词为：昇腾 <br/>

1.1 支持的产品  
Ascend 310 、 Altas 200DK

1.2 支持的版本  
CANN：5.1RC2（通过cat /usr/local/Ascend/ascend-toolkit/latest/acllib/version.info，获取版本信息）

SDK：3.0RC3（可通过cat SDK目录下的version.info查看信息）

1.3 代码目录结构与说明  
工程目录如下图所示：

```
|-------- pipeline
|           |---- crnn_ctc.pipeline            //声学模型流水线配置文件
|-------- python
|           |---- main.py                      //om测试样例
|           |---- kws_predict.py               //原模型测试样例
|           |---- run.sh                       //运行脚本
|-------- README.md
|-------- modelchange                   //转换模型相关目录
|           |---- keras2onnx.py         //h5->pb->onnx
|           |---- onnx2om.py            //onnx->om
|-------- coverModel.sh                 //转换模型脚本
```
1.4 技术实现流程  
本项目首先通过onnx软件将tf的预训练模型转化为onnx模型，然后在使用atc工具将其转化为SDK能使用的om模型。最终通过构建SDK推理pipeline，实现模型推理。

1.5 特性及适用场景  
  kws主要分为两个步骤：

  >1 构建声学模型  
  2 对模型输出进行解码，查看是否出现目标关键词

  声学模型采用CRNN-CTC,模型构建参考论文《CRNN-CTC Based Mandarin Keyword Spotting》<br/>


当存在以下情况时：音频语速过快，把“昇腾”关键字和其他非关键字快速读完、音频语速过慢，把“昇腾”关键字拆分成“昇”和“腾”读完且中间间隔时间较长、音频中杂音较多，“昇腾”语音不清晰。
会干扰音频特征的提取，可能会导致检测效果不佳。
同时由于数据集大小问题精度还有上升空间，大量的数据集训练能够在一定程度上改善这些问题。


## 2 数据集准备
本节环境依赖如下，可在非昇腾环境完成：
|软件名称    | 版本     |
|-----------|----------|
| python    | 3.9.2    |
| tensorflow | 2.6.2  |          
| numpy   | 1.22.4   |
| librosa  | 0.9.2    |

数据集处理时用到了tensorflow和librosa，tensorflow和librosa是对numpy版本都有要求，其中librosa对numpy版本是要求1.22或者更低，而过高版本的tensorflow要求numpy是1.23。

librosa安装若无法编译相关依赖，可参考下述指令在root用户下安装对应的库<br/>
```bash
apt-get install llvm-10 -y
LLVM_CONFIG=/usr/lib/llvm-10/bin/llvm-config pip install librosa
apt-get install libsndfile1 -y
apt-get install libasound2-dev libsndfile-dev
apt-get install liblzma-dev
```

>此处提供处理完成的数据集data_CTC目录([下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/kws/data_CTC.7z))，生成的索引目录data_CTC_pre_2([索引目录下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/kws/data_CTC_pre_2.7z))。请注意数据集每次预处理生成的data_CTC_pre_2索引和数据集的mod_gsc2均不同且一一对应，并会影响后续精度测试结果。 

### 2.1 下载tensorflow原作的([下载链接](https://github.com/ryuuji06/keyword-spotting/))的代码并部署
### 2.2 下载后数据集data_CTC([下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/kws/data_CTC.7z))后删除mod_gsc2文件夹内的文件，部署至原项目根目录下。

>注意在arm环境中可能存在numpy版本问题导致无法正确切分数据集，此种情况可在其他环境先处理好数据集，本例中数据集相关操作在Windows平台完成。
下载下来的源码文件keyword-spotting-main部署至推理项目根目录(keyword-spotting-main/)下

### 2.3 执行数据集切分
下载源码文件后将create_dataset.py 和 prepare_datasets.py 中的keywords由  
>keywords = ['house', 'right', 'down', 'left', 'no', 'five', 'one', 'three']

替换成
>keywords =['昇腾']

将create_dataset.py中的
>fillers = ['stop', 'bird', 'bed', 'cat', 'eight', 'go', 'wow', 'four','happy', 'marvin', 'on', 'off', 'sheila', 'zero', 'yes', 'up', 'tree', 'seven' ]

替换成
>fillers = ['上海','北京','中国','城市','记者','政策']

将下载好的数据集放在源码文件keyword-spotting-main文件夹中
将create_dataset.py中数据集路径
>source_folder = 'data\\speech_commands' # path to speech commands dataset  
target_folder = 'data\\mod_gsc' # path to created dataset


替换成下载好的数据集位置
>source_folder = 'data_CTC\\words' # path to speech commands dataset  
target_folder = 'data_CTC\\mod_gsc2' # path to created dataset

将prepare_datasets.py中数据集路径
>cmds1_path = 'data/speech_commands'  
cmds2_path = 'data/mod_gsc'  
libri_path = "data/train-clean-100/LibriSpeech/train-clean-100"  
noisefolder = 'data/MS-SNSD/noise_train'  
......  
selected_data_folder = 'data2'


替换成create_dataset.py处理后会生成的数据集位置
>cmds1_path = 'data_CTC/words'  
cmds2_path = 'data_CTC/mod_gsc2'  
libri_path = 'data_CTC/S0002'  
noisefolder = 'data_CTC/noise_train'  
......  
selected_data_folder = 'data_CTC_pre_2'


替换完成后，预处理数据集。  
>python create_dataset.py   
python prepare_datasets.py  

完成后生成如下目录结构（仅展示所需部分），保存以下文件
```
keyword-spotting-main
|-------- data_CTC_pre_2               //处理后的数据索引目录
|-------- data_CTC                    //经过处理的数据目录(子文件夹正确释放示例如下)
|           |---- words
|           |---- S0002
|           |---- noise_train
|           |---- mod_gsc2          //处理生成的文件夹且每次内容不同，原始数据集无此文件夹，自行执行数据预处理时请删除该文件夹
```
## 3 模型训练

本章节依赖同## 2数据集准备，可在同一环境完成

>此处提供训练生成的result1ctc1目录([下载链接](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/kws/result1ctc1.rar))

cd进入原项目文件夹（keyword-spotting-main）并执行以下命令
>python kws_train.py -d ../data_CTC_pre_2 -r ../result1ctc1 -m 2 --train_test_split 0.8 --epochs 50

train_test_split 是将数据集按比例区分成训练集和测试集.data_CTC_pre_2是prepare_datasets.py生成的数据集。result1ctc1是模型。
训练完成后保存h5模型和相关参数文件至result1ctc1目录（仅展示所需部分），保存这些文件
```
keyword-spotting-main
|-------- result1ctc1           //训练后的h5模型及相关参数目录
```

## 4 模型推理
本节环境依赖如下，在310或200DK环境完成：
|软件名称    | 版本     |
|-----------|----------|
| python    | 3.9.2    |
| MindX SDK | 3.0RC3   |
| tensorflow | 2.6.2  |          
| soundfile | 0.10.3   |
| numpy  | 1.22.4   |
| tf2onnx（模型准换）   | 1.12.1   |
| MagicONNX（模型准换） | 0.1.0    |

MagicONNX无法使用pip安装，请手动进行如下安装操作
arm依赖onnxsim，该项安装可能异常。
```bash
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```


### 4.1 激活310或200DK环境  
运行
```bash
source ${SDK−path}/set_env.sh
source ${ascend-toolkit-path}/set_env.sh
```
以激活环境，其中SDK-path是SDK mxVision安装路径，ascend-toolkit-path是CANN安装路径。

### 4.2 模型转换  
可直接使用相关脚本，注意该文件会尝试加载默认安装的/usr/local/Ascend/ascend-toolkit/set_env.sh以激活atc环境变量。注意依赖MagicONNX和onnxsim组件，但onnxsim在arm服务器可能存在安装问题。
>bash coverModel.sh

或手动执行：
激活环境后执行以下命令
```bash
#1 转换.h5 to .pb模型
python modelchange/keras2onnx.py

#2 转换.pb to onnx模型
python -m tf2onnx.convert --saved-model tmp_model --output model1.onnx --opset 11

#3 修正onnx模型的冗余节点输入，否则atc报错
python modelchange/onnx2om.py

#4 ATC转换onnx到OM模型
atc --framework=5 --model=model2.onnx --output=modelctc1 --input_format=ND --input_shape="input:1,958,13" --log=debug --soc_version=Ascend310
```
在modelchange目录下生成相关OM模型

>atc中使用的onnx模型以及产生的om模型提供下载链接：
[onnx模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/kws/model2.onnx)
[om模型](https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/kws/modelctc1.om)

### 4.3 SDK推理
部署模型和数据集文件至推理项目目录，最终的结构如下：

```
|-------- pipeline
|           |---- crnn_ctc.pipeline            //声学模型流水线配置文件
|-------- python
|           |---- main.py                      //om测试样例
|           |---- kws_predict.py               //原模型测试样例
|           |---- run.sh                       //运行脚本
|-------- README.md                     //本文档
|-------- modelchange                   //转换模型相关目录
|           |---- keras2onnx.py         //h5->pb->onnx
|           |---- onnx2om.py            //onnx->om
|-------- result1ctc1                   //训练后的h5模型及相关参数目录
|-------- data_CTC_pre_2               //预处理后的数据索引目录
|-------- data_CTC                    //经过预处理的数据目录
|-------- blank                     // 过长音频截取长度后的存放的临时文件夹(推理运行时生成)
```

修改run.sh中代码以指定OM精度测试，OM指定音频文件夹功能测试或原模型tf精度测试，原模型tf指定音频文件夹功能测试。
```bash
# python main.py -p "../{}"         #OM指定文件夹功能
# python kws_predict.py -p "../{}"      #原模型tf指定文件夹功能
# python kws_predict.py     #原模型tf精度
python main.py    #OM精度
```
执行bash run.sh以启动对应功能
### 4.4 结果说明
1. 功能测试  
输出shape和对应文件名，以及是否包含关键词
```
(1, 958, 13)
../blank/0067-城市-政策.wav: predict [[0, 0]],NOT Including keyword promotion "shengteng"
```

2. 精度测试
>先输出预测结果“tokens_post_p1”的值。再输出真实标签“tokens_true_p1”的值。其中1 代表关键字昇腾，0 代表其他关键字。如果真实标签和预测标签都有“昇腾”关键字，预测准确数量+1，如果都没有“昇腾”，预测准确数量+1。最后输出精度。精度 = 预测准确数量除以音频总数。

```
#原模型
......
tokens_post_p1 [[0]]
tokens_true_p1 [[0, 0, 0, 0, 0, 0, 0, 0]]
274-------------------------------
tf model accuracy: 0.9454545454545454

#om模型
(1, 958, 13)
tokens_post_p1 [[0]]
tokens_true_p1 [[0, 0, 0, 0, 0, 0, 0, 0]]
274-------------------------------
om model accuracy: 0.9418181818181818

```
精度误差小于1%
## 5 其它说明

测试精度时，音频数据长度应小于arg.audiolen，否则被截断，因此过长的音频得到结果可能会不准。arg.audiolen的长度应与om模型的输入shape相对应。