import numpy as np
def nms(dets, thresh):
    # dets:(m,5)  thresh:scaler
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4] 
    areas = (y2- y1+ 1) * (x2- x1+ 1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
        
        overlaps = w*h
         
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
         
        idx = np.where(ious<=thresh)[0]
         
        index = index[idx+1]   # because index start from 1
         
    return keep
