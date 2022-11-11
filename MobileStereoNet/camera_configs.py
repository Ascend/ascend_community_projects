# filename: camera_configs.py
import cv2
import numpy as np

left_camera_matrix = np.array([[823.3582, 0., 365.5732],
                               [0., 831.0410, 250.4912],
                               [0., 0., 1.]])
left_distortion = np.array([[0.2194, -1.1576, 0.0071, -0.0049, 0.0000]])



right_camera_matrix = np.array([[832.8658, 0., 384.2469],
                                [0., 839.1351, 244.8956],
                                [0., 0., 1.]])
right_distortion = np.array([[0.1951, -1.4400, 0.0059, -0.0017, 0.0000]])

# om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
R = np.array([[0.9990, -0.0037, -0.0442],
               [0.0023, 0.9995, -0.0304],
               [0.0443, 0.0303, 0.9986]])
# R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([76.5799, 1.0316, 4.9442]) # 平移关系向量

size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)




