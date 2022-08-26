# C++ 基于MxBase 的RefineDet图像检测样例及RefineDet的后处理模块开发

## 介绍

本开发样例是基于mxBase开发的端到端推理的C++应用程序，可在昇腾芯片上进行 RefineDet目标检测，并把可视化结果保存到本地。其中包含RefineDet的后处理模块开发。
该Sample的主要处理流程为：
Init > ReadImage >Resize > Inference >PostProcess >DeInit

## 模型转换

本项目中使用的模型是RefineDet模型，onnx模型可以直接[下载](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/47d31ca99aa641b2b220cabc9233cdb7)。下载后解包，得到`RefineDet320_VOC_final_no_nms.onnx`，使用模型转换工具ATC将onnx模型转换为om模型，模型转换工具相关介绍参考[链接](https://support.huaweicloud.com/tg-cannApplicationDev330/atlasatc_16_0005.html)

模型转换步骤如下：

1、按照2环境依赖设置环境变量，并运行以下命令

````
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
````

2、`cd`到`models`文件夹，运行

````
atc --framework=5 --model=RefineDet320_VOC_final_no_nms.onnx --output=RefineDet --input_format=NCHW --input_shape="image:1,3,320,320" --log=debug --soc_version=Ascend310 --insert_op_conf=../config/RefineDet.aippconfig --precision_mode=force_fp32
````

3、执行该命令后会在指定输出.om模型路径生成项目指定模型文件RefineDet.om`。若模型转换成功则输出：

```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
```

aipp文件配置如下：

```
aipp_op {
    related_input_rank : 0
    src_image_size_w : 320
    src_image_size_h : 320
    crop : false
    aipp_mode: static
    input_format : YUV420SP_U8
    csc_switch : true
    rbuv_swap_switch : true
    matrix_r0c0 : 256
    matrix_r0c1 : 454
    matrix_r0c2 : 0
    matrix_r1c0 : 256
    matrix_r1c1 : -88
    matrix_r1c2 : -183
    matrix_r2c0 : 256
    matrix_r2c1 : 0
    matrix_r2c2 : 359
    input_bias_0 : 0
    input_bias_1 : 128
    input_bias_2 : 128
    mean_chn_0 : 104
    mean_chn_1 : 117
    mean_chn_2 : 123
    min_chn_0 : 0.0
    min_chn_1 : 0.0
    min_chn_2 : 0.0
    var_reci_chn_0 : 1.0
    var_reci_chn_1 : 1.0
    var_reci_chn_2 : 1.0
}
```

## 编译与运行

**步骤1** 修改CMakeLists.txt文件 将set(MX_SDK_HOME \$\{SDK安装路径\}\$) 中的\${SDK安装路径}\$替换为实际的SDK安装路径

**步骤2** 设置环境变量`ASCEND_HOME` Ascend安装的路径，一般为`/usr/local/Ascend
LD_LIBRARY_PATH `指定程序运行时依赖的动态库查找路径，包括ACL，开源软件库。

```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${MX_SDK_HOME}/lib:${MX_SDK_HOME}/opensource/lib:${MX_SDK_HOME}/opensource/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64:${LD_LIBRARY_PATH}
```

**步骤3** cd到mxbase目录下，执行如下编译命令：

````
bash build.sh
````

**步骤4** 制定jpg图片进行推理，准备一张推理图片放入mxbase 目录下。eg:推理图片为test.jpg
cd 到mxbase 目录下

```
./refinedet ./test.jpg
```



