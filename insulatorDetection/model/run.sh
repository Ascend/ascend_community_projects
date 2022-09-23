
. ${SDK-path}/set_env.sh  ## 修改SDK-path为你自己的SDK安装路径
. ${ascend_toolkit_path}/set_env.sh ## 修改ascend_toolkit_path为自己ascend的ascend_toolkit路径


atc --output_type=FP32 --insert_op_conf=yolo_aipp.cfg --input_format=NCHW --framework=5 --model=./insulator.onnx --input_shape="input:1,3,416,416"  --output=./insulator   --soc_version=Ascend310  --enable_small_channel=1
