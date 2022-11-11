import numpy as np
import cv2
import math
import camera_configs
from yolov3_infer import yolov3

leftimg_path = "./snapshot/left_0.jpg"
rightimg_path = "./snapshot/right_0.jpg"

def BM(imgleft, imgright):
    stereo = cv2.StereoBM_create(numDisparities=16 * 1, blockSize=5)
    disparity = stereo.compute(imgleft, imgright)
    return disparity

def SGBM(imgleft, imgright):
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=16 * 1,
                                   blockSize=5,
                                   P1=216,
                                   P2=864,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=15,
                                   speckleWindowSize=0,
                                   speckleRange=1,
                                   preFilterCap=60,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(imgleft, imgright)
    return disparity

frame1 = cv2.imread(leftimg_path)
frame2 = cv2.imread(rightimg_path)

# 根据更正map对图片进行重构
img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
cv2.imwrite("BM_right.jpg",img2_rectified)


# 将图片置为灰度图，为StereoBM作准备 
imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)


# 根据Block Maching方法生成差异图
# disparity = BM(imgL, imgR)

# 根据SGBM/Semi-Global Block Matching方法生成差异图
disparity = SGBM(imgL, imgR)

# 将图片扩展至3d空间中，其z方向的值则为当前的距离
threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)

# 因为om模型读取要YUV格式，前面cv读取处理是BGR，我暂时没找到直接定义Image类的方法，所以重读一遍重构后的图片
coordinate = yolov3("BM_right.jpg")

for i in range(len(coordinate)):
    x = coordinate[i].x1 
    y = coordinate[i].y1 

    x = int(x)
    y = int(y)

    print('\n像素坐标 x = %d, y = %d' % (x, y))
    # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
    print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

    distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
    distance = distance / 1000.0  # mm -> m
    print("距离是：", distance, "m")
