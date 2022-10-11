
1. 从链接中下载 onnx 模型 best.onnx 至 ``python/models`` 文件夹下。

2. 将该模型转换为om模型，具体操作为： ``python/models`` 文件夹下,执行atc指令：

1)加预处理
```
atc --model=best.onnx --framework=5 --output=./yolox_pre_post --output_type=FP32 --soc_version=Ascend310  --input_shape="images:1, 3, 640, 640" --insert_op_conf=./aipp-configs/yolox_bgr.cfg
```
2)不加预处理
```
atc --model=best.onnx --framework=5 --output=./yolox_nopre_post --output_type=FP32 --soc_version=Ascend310  --input_shape="images:1, 3, 640, 640" 
```
注：两种方法区在于对之后的图片是否进行缩放，会导致验证精度不同。

若终端输出：
```
ATC start working now, please wait for a moment.
ATC run success, welcome to the next use.
W11001: Op [Slice_30] does not hit the high-priority operator information library, which might result in compromised performance.
W11001: Op [Slice_10] does not hit the high-priority operator information library, which might result in compromised performance.
W11001: Op [Slice_40] does not hit the high-priority operator information library, which might result in compromised performance.
W11001: Op [Slice_20] does not hit the high-priority operator information library, which might result in compromised performance.

```

表示命令执行成功。

## 4. 编译与运行
**步骤1** 在项目根目录执行命令：
 
```
bash build.sh  
```   

**步骤2** 放入待测图片。将一张图片放在路径``python/test_img``下，命名为 test.jpg。

**步骤3** 图片检测。在项目路径``python/Main``下运行命令：

```
python3 pre_post.py
python3 nopre_post.py
```     

命令执行成功后在目录``python/test_img``下生成检测结果文件 pre_post_bgr.jpg(nopre_post.py)，查看结果文件验证检测结果。

## 5. 精度测试

1. 下载COCO VAL 2017[验证数据集和标注文件](https://github.com/bywang2018/security-dataset)，此数据集文件夹组织形式如下图所示

```
pidray
├── annotations                                                                                 
│    ├── test_easy.json
│    ├── test_hard.json
│    ├── test_hidden.json
│    └── train.json                                                                             
├── easy
├── hard
├── hidden
└── train                                                                                                                  
```
其中，easy,hard,hidden,train中存放的都是png格式的图片

将hard文件夹和annotations中的test_hard.json文件保存在项目目录``python/test/data``下，其中将hard文件夹名改成val2017，test_hard.json改成instances_val2017.json此文件夹下的组织形式应如下图所示：

```
├── annotations                                                                                                                                                                             
│    └── instances_val2017.json                                                                             
└──val2017                                                                                                                  
```

其中val2017文件夹下应存放jpg格式的待检测图片。

2. 使用以下指令运行路径``python/test``下的文件 parse_coco.py                         
```
python3 parse_coco.py --json_file=data/annotations/instances_val2017.json --img_path=data/val2017
```              
若运行成功，会在该目录下生成文件夹ground_truth，其中包含每张图像上提取的目标框真实位置与类别的txt文件。                         
                                              
接下来将每张图的预测结果转为txt文件，并保存在同一文件夹下，其步骤如下：

3. 进入``python/Main``路径，运行命令：
```
python3 eval_pre_post.py
python3 eval_nopre_post.py
```                      

若运行成功，会在``python/test`` 路径下生成 test_pre_post(test_nopre_post) 文件夹，该目录下包含有每张图像上的检测结果的 txt 文件。

4. 在``python/test``路径下，运行命令: 
```                                                        
python3 map_calculate.py  --npu_txt_path="./test_pre_post" 
python3 map_calculate.py  --npu_txt_path="./test_nopre_post" 
``` 
若运行成功则得到最终检测精度，结果如下：

<center>
    <img src="./images/result_map.png">
    <br>
</center>
挑选的是hard中的图片验证精度，精度为31.68%与源项目精度31.82%误差为0.14%。精度对齐。


注：在pipeline中加图像预处理后验证结果与原框不同的原因为：YOLOX的图像预处理中，Resize方式为按长边缩放，而Mindx SDK默认使用dvpp的图像解码方式，没有按长边缩放的方法，因此本项目将"resizeType"属性设置为 "Resizer_KeepAspectRatio_Fit"，这样会导致精度下降。
我们同时给出了一套不加图像预处理的推理流程，见上文，不加预处理流程精度结果与源项目可以保持一致。

## 6 常见问题

### 6.1 模型转换时会警告缺slice算子

YOLOX在图像输入到模型前会进行slice操作，而ATC工具缺少这样的算子，因此会报出如图所示的警告：

<center>
    <img src="./images/warning.png">
    <br>
</center>

**解决方案：**

常规的做法是修改slice算子，具体操作可参考[安全帽检测](https://gitee.com/booyan/mindxsdk-referenceapps/tree/master/contrib/HelmetIdentification)的开源项目。

由于在本项目下是否修改算子并不影响检测结果，因此默认不做处理。

### 6.2 图片无法识别

**解决方案：**

png格式图片需要转换成jpg格式图片再进行检测。