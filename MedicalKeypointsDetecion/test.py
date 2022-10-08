from xmlrpc.client import boolean
from StreamManagerApi import *
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import cv2
import os
import math
from PIL import Image
import mindspore as ms
import mindspore.ops as ops

color2 = [(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),
          (0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),(0,0,255),(0,255,0),(255,0,0),(0,0,139),(0,69,255),]


def get_img_metas(file_name):
    img = Image.open(file_name)
    img_size = img.size

    org_width, org_height = img_size
    resize_ratio = 1280 / org_width
    if resize_ratio > 768 / org_height:
        resize_ratio = 768 / org_height

    img_metas = np.array([img_size[1], img_size[0]] +
                         [resize_ratio, resize_ratio])
    return img_metas

def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        result = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        result_person = bboxes[labels == 0, :]
    return result, result_person

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale
    
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px] - hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def mask_generate(filter, num_joint):
    class_num = [27, 5, 10, 6, 6, 4, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    idx_num =  [0, 27, 32, 42, 48, 54, 58, 60, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
    mask = np.zeros(
        (len(filter), num_joint, 2),
        dtype=np.float32
    )
    for i, index in enumerate(filter):
        for j in range(idx_num[index], idx_num[index + 1]):
            mask[i][j][0] = 1
            mask[i][j][1] = 1

    return mask

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (80, 2)
    for i in range(80):
        x_a, y_a = keypoints[i][0], keypoints[i][1]
        cv2.circle(img, (int(x_a), int(y_a)), 10, color2[i], 10)

def normalize(data, mean, std):
    # transforms.ToTensor, transforms.Normalize的numpy 实现
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    if mean.ndim == 1:
        mean = np.reshape(mean, (-1, 1, 1))
    if std.ndim == 1:
        std = np.reshape(std, (-1, 1, 1))
    _max = np.max(abs(data))
    _div = np.divide(data, 255)  # i.e. _div = data / _max
    _div = np.transpose(_div, (2, 0, 1))
    _sub = np.subtract(_div, mean)  # i.e. arrays = _div - mean
    arrays = np.divide(_sub, std)  # i.e. arrays = (_div - mean) / std
    return arrays

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/model1.pipeline", 'rb') as f1:
        pipelineStr = f1.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
        
    # Construct the input of the stream & check the input image
    dataInput = MxDataInput()
    filepath = "test.jpg"
    
    if os.path.exists(filepath) != 1:
        print("Failed to get the input picture. Please check it!")
        streamManagerApi.DestroyAllStreams()
        exit()
    
    with open(filepath, 'rb') as f:
        dataInput.data = f.read()
                    
    # Inputs data to a specified stream based on streamName.
    streamName1 = b'model1'
    inPluginId = 0
    uniqueId = streamManagerApi.SendData(streamName1, inPluginId, dataInput)

    #send image data
    metas = get_img_metas(filepath).astype(np.float32).tobytes()

    key = b'appsrc1'
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    visionVec.visionData.deviceId = 0
    visionVec.visionData.memType = 0
    visionVec.visionData.dataStr = metas
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec = InProtobufVector()
    protobufVec.push_back(protobuf)

    inPluginId1 = 1
    uniqueId1 = streamManagerApi.SendProtobuf(streamName1, b'appsrc1', protobufVec)

    if uniqueId1 < 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keys = b"mxpi_tensorinfer0"
    for key in keys:
        keyVec.push_back(keys)
    infer_result = streamManagerApi.GetProtobuf(streamName1, 0, keyVec)
    # print the infer result
    
    if infer_result.size() == 0:
        print("infer_result is null")
        exit()

    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            infer_result[0].errorCode, infer_result[0].data.decode()))
        exit()

    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[0].messageBuf)
    pre_mask = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[2].dataStr, dtype = boolean).reshape((1,80000,1))
    pre_label = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype = np.uint32).reshape((1,80000,1))
    pre_bbox = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float16).reshape((1,80000,5))

    bbox_squee = np.squeeze(pre_bbox.reshape(80000, 5))
    label_squee = np.squeeze(pre_label.reshape(80000, 1))
    mask_squee = np.squeeze(pre_mask.reshape(80000, 1))

    all_bboxes_tmp_mask = bbox_squee[mask_squee, :]
    all_labels_tmp_mask = label_squee[mask_squee]

    if all_bboxes_tmp_mask.shape[0] > 128:
        inds = np.argsort(-all_bboxes_tmp_mask[:, -1]) # 返回降序排列索引值
        inds = inds[:128]
        all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
        all_labels_tmp_mask = all_labels_tmp_mask[inds]

    outputs = []
    outputs_tmp, out_person = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, 81)
    outputs.append(outputs_tmp)

    img = cv2.imread(filepath)
    box_person = (int(out_person[0][0]), int(out_person[0][1])) , (int(out_person[0][2]), int(out_person[0][3]))
    

    ret2 = streamManagerApi.InitManager()
    if ret2 != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/model2.pipeline", 'rb') as f2:
        pipelineStr2 = f2.read()
    ret2 = streamManagerApi.CreateMultipleStreams(pipelineStr2)
    if ret2 != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    image_bgr = cv2.imread(filepath)
    image = image_bgr[:, :, [2, 1, 0]]

    input = []
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = np.transpose(img / 255., (2, 0, 1))
    input.append(img_tensor)

    center, scale = box_to_center_scale(box_person, 288, 384) 
    image_pose = image.copy() 
    # model 2 preprocess
    trans = get_affine_transform(center, scale, 0, [288, 384])
    model_input = cv2.warpAffine(
        image_pose,
        trans,
        (288, 384),
        flags = cv2.INTER_LINEAR)

    mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype = np.float32)
    data_test = normalize(model_input, mean, std)
    data_test = data_test.astype('float32')
    data_test = np.reshape(data_test, (1,3,384,288))
    print("model_input_gai", data_test.shape, data_test)

    tensors = data_test
    streamName2 = b'test_model2'
    inPluginId = 0

    tensorPackageList = MxpiDataType.MxpiTensorPackageList()
    tensorPackage = tensorPackageList.tensorPackageVec.add()
    print(tensors.shape)
    array_bytes = tensors.tobytes()
    dataInput = MxDataInput()
    dataInput.data = array_bytes
    tensorVec = tensorPackage.tensorVec.add()
    tensorVec.deviceId = 0
    tensorVec.memType = 0
    for i in tensors.shape:
        tensorVec.tensorShape.append(i)
    tensorVec.dataStr = dataInput.data
    tensorVec.tensorDataSize = len(array_bytes)

    key = "appsrc0".encode('utf-8')
    protobufVec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensorPackageList.SerializeToString()
    protobufVec.push_back(protobuf)

    ret = streamManagerApi.SendProtobuf(streamName2, inPluginId, protobufVec)
    if ret != 0:
        print("Failed to send data to stream.")
        exit()

    keyVec = StringVector()
    keys2 = b"mxpi_tensorinfer0"
    for key in keys2:
        keyVec.push_back(keys)
    infer_result = streamManagerApi.GetProtobuf(streamName2, 0, keyVec)
    tensorList = MxpiDataType.MxpiTensorPackageList()
    tensorList.ParseFromString(infer_result[0].messageBuf)
    keypoint_outputs = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float32).reshape((1,80,96,72))
    cls_outputs = np.frombuffer(tensorList.tensorPackageVec[0].tensorVec[1].dataStr, dtype = np.float32).reshape((1,23))

    #post_process
    if isinstance(keypoint_outputs, list):
        kp_output = keypoint_outputs[-1]
    else:
        kp_output = keypoint_outputs
    
    preds, maxvals = get_final_preds(
            kp_output,
            np.asarray([center]),
            np.asarray([scale]))
    filter = np.argmax(cls_outputs, axis = 1)
    mask = mask_generate(filter, 80)
    
    preds_mask = preds * mask        
    image_bgr = cv2.imread(filepath)
    if len(preds_mask) >= 1:
        for kpt in preds_mask:
            draw_pose(kpt, image_bgr)  # draw the poses

    save_path = 'output.jpg'
    cv2.imwrite(save_path, image_bgr)
    print('the result image has been saved as {}'.format(save_path)) 

    # destroy streams
    streamManagerApi.DestroyAllStreams()