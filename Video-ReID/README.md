# MindXSDK 视频行人重识别

## 1 简介
本开发样例基于MindX SDK实现了对图片和视频流的行人重识别（Person Re-identification, ReID），支持检索给定照片、视频中的行人ID。其主要流程为：    
- 构建行人特征库：将底库图片调整大小，利用目标检测模型YOLOv3推理，检测图片中的行人，检测结果经过抠图与调整大小，再利用OSNet模型提取图片中每个行人的特征向量并保存，特征向量用于与后续的待查询图片或者视频中的行人作比较。
- 对于查询图片或视频帧：利用目标检测模型YOLOv3推理，检测图片中的行人，检测结果经过抠图与调整大小，再利用OSNet模型提取图片中每个行人的特征向量。    
- 行人检索：将查询图片中行人的特征向量与底库中的特征向量做比较，为每个查询图片中的行人检索最有可能的ID，通过识别框和文字信息进行可视化标记。
- 如果输入是图片，最终得到标记过的图片文件，如果是视频流，得到标记过的H264格式视频文件

## 2 目录结构
本工程名称为Video-ReID，工程目录如下图所示：
```
video-ReID
|---- config
|   |   |---- coco.names
|   |   |---- yolov3_tf_bs1_fp16.cfg
|---- data
|   |---- gallery                       // 行人底库图片文件夹
|   |---- query                      
|       |---- images					// 查询场景图片文件夹
|       |---- videos				    // 查询场景视频文件夹,也可根据推流时实际环境配置存放位置
|---- models                            // 目标检测、OSNet模型与配置文件夹
|   |---- OSNet
|   |   |   |---- aipp.config
|---- YOLOv3
|   |   |   |---- tf_aipp.config
|---- pipeline                          // 流水线配置文件夹
|   |   |---- image.pipeline
|   |   |---- gallery.pipeline
|   |   |---- video.pipeline
|---- plugins                           // 自定义插件目录
|   |---- PluginFeatureMatch
|---- output                            // 结果保存文件夹
|   |---- gallery
|   |---- query                      
|       |---- images				
|       |---- videos                    //需自行创建        
|---- image.py
|---- gallery.py
|---- video.py
|---- run.sh
|---- README.md   
```
> 由于无法在Gitee上创建空文件夹，请按照该工程目录，自行创建output文件夹、data文件夹与其内部的文件夹 

## 3 依赖
| 软件名称 | 版本   |
| :--------: | :------: |
|ubuntu 18.04|18.04.1 LTS   |
|CANN|5.0.4|
|MindX SDK|2.0.4|
|Python| 3.9.2|
|numpy | 1.21.0 |
|opencv_python|4.5.2|

- 设置环境变量（请确认install_path路径是否正确）
```
#执行如下命令
. ${SDK-path}/set_env.sh
. ${ascend_toolkit_path}/set_env.sh
```

请注意MindX SDK使用python版本为3.9.2，如出现无法找到python对应lib库请在root下安装python3.9开发库 
```
apt-get install libpython3.9
```

- 推理中涉及到第三方软件依赖如下表所示。

| 依赖软件 | 版本       | 说明                             | 使用教程                                                     |
| -------- | ---------- | -------------------------------- | ------------------------------------------------------------ |
| live555  | 1.10       | 实现视频转 rstp 进行推流         | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/Live555离线视频转RTSP说明文档.md) |
| ffmpeg   | 2022-06-27 | 实现 mp4 格式视频转为264格式视频 | [链接](https://gitee.com/ascend/mindxsdk-referenceapps/blob/master/docs/参考资料/pc端ffmpeg安装教程.md) | 

## 4 模型转换
行人重识别先采用了yolov3模型将图片中的行人检测出来，然后利用OsNet模型获取行人的特征向量。由于yolov3模型和OsNet模型分别是基于Pytorch和Tensorflow的深度模型，我们需要借助ATC工具分别将其转换成对应的.om模型。

### 4.1 yolov3的模型转换：  

**步骤1** 获取yolov3的原始模型(.pb文件)和相应的配置文件(.cfg文件)  
&ensp;&ensp;&ensp;&ensp;&ensp; [原始模型下载链接](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/ba2a4c054a094ef595da288ecbc7d7b4) 

**步骤2** 将获取到的yolov3模型.pb文件和.cfg文件存放至：“项目所在目录/models/YOLOv3/”  

**步骤3** .om模型转换  
进入“项目所在目录/models/YOLOv3”  

- 使用ATC将.pb文件转成为.om文件
```
atc --model=yolov3_tf.pb --framework=3 --output=yolov3 --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0" --log=info --insert_op_conf=tf_aipp.cfg
```
- 执行完模型转换脚本后，若提示如下信息说明模型转换成功，可以在该路径下找到名为yolov3.om模型文件。
（可以通过修改output参数来重命名这个.om文件）
```
ATC run success, welcome to the next use.
```  

### 4.2 OSNet的模型转换

#### 4.2.1 模型概述  
&ensp;&ensp;&ensp;&ensp;&ensp; [OSNet论文地址](https://arxiv.org/pdf/1905.00953.pdf)
&ensp;&ensp;&ensp;&ensp;&ensp; [OSNet代码地址](https://github.com/KaiyangZhou/deep-person-reid)

#### 4.2.2 模型转换步骤

**步骤1** 从ModelZoo源码包中获取OSNet的onnx模型文件(osnet_x1_0.onnx) 
&ensp;&ensp;&ensp;&ensp;&ensp; [权重文件源码包下载链接](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/43a754e306c6461d86dafced5046121f) 

**步骤2** 将获取到的onnx模型存放至：“项目所在目录/models/OSNet/”  

**步骤3** .om模型转换   
进入“项目所在目录/models/OSNet”  

- 使用ATC将.onnx文件转成为.om文件
```
atc --framework=5 --model=./osnet_x1_0.onnx --input_format=NCHW --insert_op_conf=./aipp.config --input_shape="image:-1,3,256,128" --dynamic_batch_size="1,2,3,4,5,6,7,8" --output=osnet --log=debug --soc_version=Ascend310
// dynamic参数为支持的动态batchsize,可根据实际图片中可能出现的行人数目更改
```
- 执行完模型转换脚本后，若提示如下信息说明模型转换成功，可以在该路径下找到名为yolov3.om模型文件。
（可以通过修改output参数来重命名这个.om文件）
```
ATC run success, welcome to the next use.
```  
经过上述操作，可以在“项目所在目录/models”的子目录下找到yolov3.om模型和osnet.om模型，模型转换操作已全部完成

### 4.3 参考链接
> 模型转换使用了ATC工具，如需更多信息请参考：[ATC工具使用指南-快速入门](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html)  
> Yolov3模型转换的参考链接：[ATC YOLOv3(FP16)](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/ba2a4c054a094ef595da288ecbc7d7b4)  
> OSNet模型转换的参考链接：[OSNet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/OSNet)  

## 5 准备

### 5.1 数据

为适配网络输入以及性能要求，建议输入图片或视频流长宽比接近1：1，图像长宽均需为偶数，待检测行人范围像素面积大于5120且宽度大于32像素，长度大于16像素,且图像后缀为jpg/JPG。
涉及文件夹  
> “项目所在目录/data/gallery”：用于存放制作行人底库的场景图片 
建议针对待检测的行人单人正面，侧面，背面各采集图片作为其底库。图片中行人清晰醒目且身体完整。

> “项目所在目录/data/query/images”：用于存放待查询行人图片
可检测行人正面，侧面，背面。要求图片中行人清晰醒目且身体尽量完整。

> “项目所在目录/data/query/videos”：用于存放待查询行人视频
可检测行人正面，侧面，背面。要求视频帧中行人清晰醒目且身体尽量完整。

### 5.2 编译插件

项目需要用到自定义插件进行特征匹配与重标定，自定义插件源码目录为“项目所在目录/plugins/PluginFeatureMatch”

#### 5.2.1 编译插件
```
> bash build.sh
```
编译后会在${MX_SDK_HOME}/lib/plugins/目录下生成libmxpi_featurematch.so文件

#### 5.2.2 为编译获得的.so文件授予640权限.

编译参考[插件编译指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100234263/21d24289)

### 5.3 视频推流
按照第 3 小结软件依赖安装 live555 和 ffmpeg，按照 Live555离线视频转RTSP说明文档 将 mp4 视频转换为 h264 格式。并将生成的 264 格式的视频上传到 live/mediaServer 目录下，然后修改 项目所在目录/pipeline 目录下的 video.pipeline 文件中 mxpi_rtspsrc0 的内容。
```
        "mxpi_rtspsrc0": {
            "props": {
                "rtspUrl":"rtsp://xxx.xxx.xxx.xxx:xxxx/xxx.264",      // 修改为自己开发板的地址和文件名
                "channelId": "0",
                "timeout": "30"
            },
            "factory": "mxpi_rtspsrc",
            "next": "mxpi_videodecoder0"
        },
```

### 5.4 适用场景

项目适用于大部分行人目标较完整且醒目可见的场景，对于行人不完整程度高，或者行人底库中缺少同一行人对应机位的参照时，检测框或行人重标定可能会出现误差，请根据实际应用进行数据选择和处理。
对于视频，每帧的处理时长约为200ms，建议拉流视频帧率在5左右，若出现内存不足等情况请适当降低帧率。

## 6 测试

### 6.1 获取om模型
```
步骤详见4： 模型转换
```
### 6.2 准备
```
步骤详见5： 准备
```
### 6.3 配置pipeline  
根据所需场景，配置pipeline文件，调整路径参数等。
```
    # 配置mxpi_tensorinfer插件的yolov3.om模型加载路径（三个pipeline均需配置）
    "mxpi_tensorinfer0": {
        "props": {
            "dataSource": "mxpi_imageresize0",
            "modelPath": "models/YOLOv3/yolov3.om(这里根据你的命名或路径进行更改)"
        },
        "factory": "mxpi_tensorinfer",
        "next": "mxpi_objectpostprocessor0"
        },
    # 配置mxpi_objectpostprocessor插件的yolov3.cfg配置文件加载路径以及SDN的安装路径（三个pipeline均需配置）
    "mxpi_objectpostprocessor0": {
       "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath": "config/yolov3_tf_bs1_fp16.cfg(这里根据你的命名或路径进行更改)",
                "labelPath": "config/coco.names",
                "postProcessLibPath": "libyolov3postprocess.so"
            },
            "factory": "mxpi_objectpostprocessor",
            "next": "mxpi_objectselector0"
    },
    # 配置mxpi_tensorinfer插件的OsNet.om模型加载路径（三个pipeline均需配置）
    "mxpi_tensorinfer1": {
            "props": {
                "dataSource": "mxpi_imagecrop0",
                "dynamicStrategy": "Upper",
                "modelPath": "models/OSNet/osnet.om",
                "waitingTime": "1"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_featurematch0"
        },

```
### 6.4 执行

#### 6.4.1 构建行人特征库
```
bash run.sh gallery
```
执行成功后会打印单张图片处理耗时并在output/gallery目录下生成.txt标签文件和.bin特征向量存储文件。

#### 6.4.2 图片查询
```
bash run.sh image
```
执行成功后会打印单张图片处理耗时并在output/query/images目录下生成标记了Reid目标的图片输出。
经测试，端到端处理耗时在200ms以内。

#### 6.4.3 视频查询
```
bash run.sh video 60 # 60s为处理视频流时长参数，可根据实际情况修改
```
执行成功后会在output/query/videos目录下获得标记了Reid目标的h264格式视频文件。

### 6.5 查看结果  
执行命令后，可在“项目所在目录/output”路径下查看结果。


## 7 参考链接
> 特定行人检索：[Person Search Demo](https://github.com/KaiyangZhou/deep-person-reid)  


## 8 Q&A
### 8.1 在运行查询脚本时出现"[DVPP: image width out of range] The crop width of image No.[0] is out of range [32,4096]"
> 这里的错误是因为yolov3模型检测到的目标过小，调大“mxpi_objectselector0”插件的MinArea参数或者更新“项目所在目录/config/yolov3_tf_bs1_fp16.cfg”文件，将OBJECTNESS_THRESH适度调大可解决该问题

### 8.2 运行video.py时出现"[6003][stream change state fail] create stream(queryVideoProcess) failed."
> 可能是因为video.pipeline中filelink插件的保存路径文件夹未创建，请手动创建(output/query/videos).

### 8.3 运行脚本时出现 streamInstance GetResult return nullptr.
> 可能是因为图像/视频里没有检测到行人，请更换有显著行人的图像输入。

### 8.4 视频处理时出现 "[Error code unknown] Failed to send frame.","[DVPP: decode H264 or H265 fail] Decode video failed.",且在销毁stream时出现 “Failed to destroy vdec channel.", "Failed to destroy stream:queryVideoProcess"或出现 "Malloc device memory failed."
> 此错误可能是因为视频帧率过高，请适当降低.264视频帧率。
