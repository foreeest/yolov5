# 实现三维位姿估计

import cv2  
import numpy as np  
import math
  
# 需要标定, realsense L515 RGB 1080 * 1920  
# LIMBUS_R_MM = 6  
# FOCAL_LEN_X_PX = 1352.7  # 4 param in slam book
# FOCAL_LEN_Y_PX = 1360.6  
# FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2  
# PRIN_POINT = np.array([979.5840, 552.1356], dtype=np.float64)  

# rs-sense-control 获取 realsense L515 BGR8 640 x 480
LIMBUS_R_MM = 6 # 上次量的多少来着？ 
FOCAL_LEN_X_PX = 596.061
FOCAL_LEN_Y_PX = 596.014 
FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2  
PRIN_POINT = np.array([331.297, 244.254], dtype=np.float64) 

# for filtering the estimated pos
NFILTER = 1 # a batch -> how many point used to estimate one point; set to 1 so no filtering
posBuf = []
dirBuf = []
counter = 0

# for eyeball centre estimation
NEYEBALL = 30
cposBuf = []
cdirBuf = []
ccounter = 0
  
def ellipse_to_limbus(x, y, w, h, angle, limbus_switch=True, mode=0):  
    """
    caculate 3d pose
    :param x: x-position
    :param y: y-position
    :param w: width of pupil, maj_axis of ellipse
    :param h: height of pupil, min_axis of ellipse
    :param angle: rotate of ellipse
    :return: 3 pos and 3 angle, will return false if input invalid
    """

    global counter, ccounter

    if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
        # output_file = "bug_log.txt"
        # with open(output_file, "a+") as file:
        #     file.write(f"get some NaN in Geometry module\n")
        #     file.write(f"{x} {y} {w} {h}\n")
        return [], False

    # 检验无效数据
    if w <= 0 or h <= 0 or w < h:
        # output_file = "bug_log.txt"
        # with open(output_file, "a+") as file:
        #     file.write(f"invalid w or h in Geometry module\n")
        #     file.write(f"{w} {h}\n")
        return [], False
    
    # 转换到毫米空间  
    iris_z_mm = (LIMBUS_R_MM * 2 * FOCAL_LEN_Z_PX) / w  
    iris_x_mm = -iris_z_mm * (x - PRIN_POINT[0]) / FOCAL_LEN_X_PX  
    iris_y_mm = iris_z_mm * (y - PRIN_POINT[1]) / FOCAL_LEN_Y_PX  
  
    # 构建三维点  
    limbus_center = np.array([iris_x_mm, iris_y_mm, iris_z_mm], dtype=np.float64)  
    
    # 角度转换为弧度  
    psi = math.pi / 180.0 * (angle + 90)  # z-axis rotation (radians)  
    # if h / w > 1:
    #     print(f"h is {h} and w is {w}")
    tht = math.acos(h / w)   # y-axis rotation (radians)  

    if limbus_switch:  # 什么时候true
        tht = -tht  # ambiguous acos, so sometimes switch limbus  

    # 计算limbus normal  
    limb_normal = np.array([  
        math.sin(tht) * math.cos(psi),  
        -math.sin(tht) * math.sin(psi),  
        -math.cos(tht)  
    ])  

    # 校正弱透视  
    x_correction = math.atan2(-iris_y_mm, iris_z_mm)  
    y_correction = math.atan2(iris_x_mm, iris_z_mm)  
  

    # 创建旋转矩阵（使用Rodrigues公式，但这里直接构建）  
    Ry = np.array([  
        [math.cos(y_correction), 0, math.sin(y_correction)],  
        [0, 1, 0],  
        [-math.sin(y_correction), 0, math.cos(y_correction)]  
    ])  

    Rx = np.array([  
        [1, 0, 0],  
        [0, math.cos(x_correction), -math.sin(x_correction)],  
        [0, math.sin(x_correction), math.cos(x_correction)]  
    ])  

    # 应用旋转  
    limb_normal = np.dot(Ry, limb_normal)  
    limb_normal = np.dot(Rx, limb_normal)  


    if mode == 0:
        # 滤波算法：取中位数
        # TODO 改成限制帧间移动距离
        counter = (counter + 1) % NFILTER
        posBuf.append(limbus_center)
        dirBuf.append(limb_normal)

        #return [limbus_center, limb_normal], True
        if counter == NFILTER - 1:
            coords_array = np.array(posBuf)
            pos_medians = np.median(coords_array, axis=0)

            coords_array = np.array(dirBuf)
            dir_medians = np.median(coords_array, axis=0)
            # TODO get centre 

            # 写出, 格式为: 
            # x1 y1 z1
            # x2 y2 z2
            # ...

            # output_file = "raw_point.txt"
            # with open(output_file, "a+") as file:
            #     for pos in posBuf:
            #         file.write(f"{pos[0]} {pos[1]} {pos[2]}\n")

            posBuf.clear()
            dirBuf.clear()
            return [pos_medians, dir_medians], True
        else:
            return [], False
    elif mode == 1:
        ccounter = (ccounter + 1) % NEYEBALL
        cposBuf.append(limbus_center)
        cdirBuf.append(limb_normal)

        #return [limbus_center, limb_normal], True
        if ccounter == NEYEBALL - 1:
            points = np.array(cposBuf)
            dirs = np.array(cdirBuf)
            pos = nearest_intersection(points, dirs)
            cposBuf.clear()
            cdirBuf.clear()

            data = np.array(pos)
            flattened_pos = data.flatten()
            return [flattened_pos, []], True
        else:
            return [], False

# TODO 用realsense提供的深度进行计算

# 从直线簇计算眼球中心1
def nearest_intersection(points, dirs):
    """
    :param points: (N, 3) array of points on the lines
    :param dirs: (N, 3) array of unit direction vectors
    :returns: (3,) array of intersection point
    """
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),
        rcond=None
    )[0]

# 从直线簇计算眼球中心2  
def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function 
    returns the least squares intersection of the N
    lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    # generate all line direction vectors 
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1 

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p