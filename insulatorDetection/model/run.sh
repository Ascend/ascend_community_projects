

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.9.2/bin:${install_path}/arm64-linux/atc/ccec_compiler/bin:${install_path}/arm64-linux/atc/bin:$PATH
export PYTHONPATH=${install_path}/arm64-linux/atc/python/site-packages:${install_path}/arm64-linux/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/arm64-linux/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path}/arm64-linux/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

atc --output_type=FP32 --insert_op_conf=yolo_aipp.cfg --input_format=NCHW --framework=5 --model=./easy1.onnx --input_shape="input:1,3,416,416"  --output=./insulator   --soc_version=Ascend310  --enable_small_channel=1
