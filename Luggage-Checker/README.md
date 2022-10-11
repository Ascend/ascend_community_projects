
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
