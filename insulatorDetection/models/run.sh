

atc --output_type=FP32 --insert_op_conf=yolo_aipp.cfg --input_format=NCHW --framework=5 --model=./insulator.onnx --input_shape="input:1,3,416,416"  --output=./insulator   --soc_version=Ascend310  --enable_small_channel=1
