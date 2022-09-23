# insualtorDetection
## 1.介绍
本开发样例基于mindSDK开发，在昇腾芯片上对输入的图片进行电力绝缘子的检测，并将检测结果进行保存。


### 1.1 支持的产品
本项目以昇腾Atlas310卡为主要的硬件平台。

### 1.2 支持的版本
支持的SDK版本为 2.0.4, CANN 版本为 5.0.4。

### 1.3 软件方案介绍
基于mindSDK的电力绝缘子检测模型的推理流程为：<br>
<br>
待检测图片通过appsrc插件输入，然后使用mxpi_imagedecoder将图片进行解码，再通过图像缩放插件mxpi_imageresize将图片缩放至合适的分辨率，缩放后的图像输入模型推理插件mxpi_tensorinfer得到输出，随后将得到的输出输入后处理插件mxpi_objectpostprocessor进行后处理，最后将结果输出给appsink完成整个pipeline流程，最后在外部使用python对得到的结果进行绘制完成可视化显示，本系统的各模块及功能如表所示:
<br><br>
表1 系统方案各功能模块描述:
|序号|子系统|功能描述|
|-----|-----|-----|
|1|图片输入|获取jpg格式图片
|2|图片解码|解码图片为YUV420p
|3|图片缩放|将图片缩放到合适的分辨率
|4|模型推理|对输入的张量进行推理
|5|电力绝缘子检测后处理|对模型推理输出进行计算得到检测框
|6|结果可视化|将电力绝缘子检测结果保存为可视化图片
<br>
### 1.4 代码目录结构与说明
项目名为insualtorDetection，项目目录如图所示

``` 
│  README.md
├─python
│      colorlist.txt
│      visualize.py
│      main.py
│
├─image    -- 自行创建，存放检测图片
│     
│
├─plugins
│      libyolov3postprocess.
│
├─model
│      label.names
│      yolo.cfg  
│      run.sh
│      yolo_aipp.cfg
│
├─test
│      map_calculate.py
│      parse_COCO.py  
│      testmain.py
│
├─pipeline
│      detect.pipeline
```

## 2. 环境依赖
<br>

### 2.1 环境变量
<br>
运行模型前要设置环境变量，需要运行的命令已经写进shell脚本,请自行修改bash脚本中的SDK_PATH和ascend_toolkit_path
<br>

### 2.2 软件依赖

|依赖软件|版本|
|-----|-----|
CANN|20.4.0
python|3.9.2
MINDX_SDK|2.0.4
opencv-python|4.5.3
numpy|1.21.2
<br>

## 3.模型转换
本工程原型是pytorch模型，需要使用atc工具转换为om模型，模型权重文件已上传至
https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/ascend_community_projects/Insulator_detection/insulator.onnx
请点击下载,将下载好的模型放到model文件夹下,随后执行脚本
```
bash run.sh
```

## 4.编译运行
<br>

### 4.1 获取测试图片
<br>
新建一个data文件夹，将需要测试的jpg图片放入该文件夹。
<br> 
<br>

### 4.2 运行推理工程
进入python目录，打开main.py，其中有个FILENAME为输入的图片路径，RESULTNAME为输出的图片路径，将其修改为自己需要的路径。执行python文件

```
python main.py
```
查看图片检测结果是否成功

## 5.评估精度和FPS
<br>
首先在test目录下创建dataset文件夹，把要测试的coco数据集的JPGIMAGES和json放到该文件夹下。随后运行parse_COCO.py，然后运行testmain获取数据集，最后再运行map_calculate获取精度和FPS，精度结果保存在output文件夹下







