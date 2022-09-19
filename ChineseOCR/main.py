import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn, StringVector
import json
import numpy as np
import os
import glob

if __name__ == '__main__':
    # init stream manager
    pipeline_path = "chineseocr.pipeline"
    streamName = b'chineseocr'
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(pipeline_path, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    inplugin_id = 0
    # Construct the input of the stream
    data_input = MxDataInput()
    # img_path = "1389.jpg"
    output_path = os.path.dirname(os.path.realpath(__file__))
    output_exer = os.path.join(output_path, 'output')
    if not os.path.exists(output_exer):
        os.makedirs(output_exer)
    for index, img_path in enumerate(glob.glob(os.path.join(output_path, '6/*.jpg'))):
        text_label = img_path.replace('jpg', 'txt')
        with open(img_path, 'rb') as fp:
            data_input.data = fp.read()
        unique_id = stream_manager_api.SendData(streamName, b'appsrc0', data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_textgenerationpostprocessor0')
        infer_result = stream_manager_api.GetProtobuf(streamName, inplugin_id, key_vec)
        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()
        result = MxpiDataType.MxpiTextsInfoList()
        result.ParseFromString(infer_result[0].messageBuf)
        content_pic = str(result.textsInfoVec[0].text)
        print(content_pic[2:-2])
        with open(text_label, 'r', encoding='utf-8') as f:
            content = ""
            for i in f.readlines():
                content += i.strip()
            # content_pic=''
            # for i in res:
            #     content_pic+=i['text'].strip()
            # content_pic=res['text']
            with open(os.path.join(output_exer, f'{index}.txt'), 'w', encoding='utf-8') as wf:
                wf.write(content)
            with open(os.path.join(output_exer, f'{index}_pic.txt'), 'w', encoding='utf-8') as wf:
                wf.write(content_pic[2:-2])
    # with open(img_path, 'rb') as f:
    #     data_input.data = f.read()
    # inplugin_id = 0
    # send data to stream
    # unique_id = stream_manager_api.SendData(streamName, b'appsrc0', data_input)
    # if unique_id < 0:
    #     print("Failed to send data to stream.")
    #     exit()
    # inplugin_key = 1
    # Obtain the inference result by specifying streamName and uniqueId.
    # key_file = open("./1389.txt", 'r')
    # key_dict = []
    # for key in key_file.readlines():
    #     key_dict.append(key.strip())
    # mxpiTextsInfoList_key = MxpiDataType.MxpiTextsInfoList()
    # textsInfoVec_key = mxpiTextsInfoList_key.textsInfoVec.add()
    # for key in key_dict:
    #     textsInfoVec_key.text.append(key)
    # key1 = b'appsrc1'
    # protobuf_vec = InProtobufVector()
    # protobuf_key = MxProtobufIn()
    # protobuf_key.key = key1
    # protobuf_key.type = b'MxTools.MxpiTextsInfoList'
    # protobuf_key.protobuf = mxpiTextsInfoList_key.SerializeToString()
    # protobuf_vec.push_back(protobuf_key)
    # print("yes")
    # unique_id = stream_manager_api.SendProtobuf(streamName, inplugin_key, protobuf_vec)

    # if unique_id < 0:
    #     print("Failed to send data to stream.")
    #     exit()
    # key_vec = StringVector()
    # key_vec.push_back(b'mxpi_textgenerationpostprocessor0')
    # infer_result = stream_manager_api.GetProtobuf(streamName, inplugin_id, key_vec)
    # if infer_result.size() == 0:
    #     print("infer_result is null")
    #     exit()
    # if infer_result[0].errorCode != 0:
    #     print("GetProtobuf error. errorCode=%d" % (
    #         infer_result[0].errorCode))
    #     exit()
    # result = MxpiDataType.MxpiTextsInfoList()
    # result.ParseFromString(infer_result[0].messageBuf)
    # print(result.textsInfoVec[0].text)
    # destroy streams
    stream_manager_api.DestroyAllStreams()
