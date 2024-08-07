# 实现三维位姿估计

import cv2  
import numpy as np  
import math
  
# 需要标定, realsense L515 RGB 1080 * 1920，这跟直接获取还差些  
LIMBUS_R_MM = 6.3  
# 默认内参  
FOCAL_LEN_X_PX = 1352.7  # 4 param in slam book
FOCAL_LEN_Y_PX = 1360.6  
FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2  
PRIN_POINT = np.array([979.5840, 552.1356], dtype=np.float64)  

# rs-sensor-control 获取 realsense L515 BGR8 640 x 480
# LIMBUS_R_MM = 6 # 上次量的多少来着？ 
# FOCAL_LEN_X_PX = 596.061
# FOCAL_LEN_Y_PX = 596.014 
# FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2  
# PRIN_POINT = np.array([331.297, 244.254], dtype=np.float64) 

def set_intrinsic(img_w, img_h):
    global FOCAL_LEN_X_PX, FOCAL_LEN_Y_PX, FOCAL_LEN_Z_PX, PRIN_POINT
    if img_h == 480 and img_w == 640:
        FOCAL_LEN_X_PX = 596.061
        FOCAL_LEN_Y_PX = 596.014 
        FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2  
        PRIN_POINT = np.array([331.297, 244.254], dtype=np.float64) 
    elif img_h == 1080 and img_w == 1920:
        FOCAL_LEN_X_PX = 1352.7
        FOCAL_LEN_Y_PX = 1360.6   
        FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2  
        PRIN_POINT = np.array([979.5840, 552.1356], dtype=np.float64) 


# for filtering the estimated pos
NFILTER = 1 # a batch -> how many point used to estimate one point
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
    if w <= 0 or h <= 0 :
        # output_file = "bug_log.txt"
        # with open(output_file, "a+") as file:
        #     file.write(f"invalid w or h in Geometry module\n")
        #     file.write(f"{w} {h}\n")
        return [], False
    
    # 转换到毫米空间  
    iris_z_mm = (LIMBUS_R_MM * 2 * FOCAL_LEN_Z_PX) / w  
    iris_x_mm = -iris_z_mm * (x - PRIN_POINT[0]) / FOCAL_LEN_X_PX  
    iris_y_mm = iris_z_mm * (y - PRIN_POINT[1]) / FOCAL_LEN_Y_PX  

    # print(f"1{x} {y}")
    # print(f"2{iris_x_mm} {iris_y_mm}")
  
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


# 无效值直接返回5个0
def fit_rotated_ellipse_ransac(data,iter=50,sample_num=10,offset=80.0):

    count_max = 0
    effective_sample = None

    for i in range(iter):
        sample = np.random.choice(len(data), sample_num, replace=False)

        xs = data[sample][:,0].reshape(-1,1)
        ys = data[sample][:,1].reshape(-1,1)
        # float -> float32
        J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float32))) )
        Y = np.mat(-1*xs**2)
        if np.linalg.det(J.T * J) == 0:
            return 0, 0, 0, 0, 0
        P = (J.T * J).I * J.T * Y

        # fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
        a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
        ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

        # threshold 
        ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])

        if(len(ran_sample) > count_max):
            count_max = len(ran_sample) 
            effective_sample = ran_sample

    return fit_rotated_ellipse(effective_sample)


def fit_rotated_ellipse(data):

    xs = data[:,0].reshape(-1,1) 
    ys = data[:,1].reshape(-1,1)
    # float -> float32
    J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float32))) ) # 雅各比矩阵
    Y = np.mat(-1*xs**2) # 目标矩阵
    if np.linalg.det(J.T * J) == 0:
        return 0, 0, 0, 0, 0
    P = (J.T * J).I * J.T * Y

    a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0]; # 计算椭圆6参数
    if a != c:
        theta = 0.5* np.arctan(b/(a-c))  # 保证 a != c
    else:
        return 0, 0, 0, 0, 0
    
    cx = (2*c*d - b*e)/(b**2-4*a*c)
    cy = (2*a*e - b*d)/(b**2-4*a*c)

    cu = a*cx**2 + b*cx*cy + c*cy**2 -f
    w2 = cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2)
    h2 = cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2)
    if w2 <= 0 or h2 <= 0: # 防止开根负数
        return 0, 0, 0, 0, 0
    w= np.sqrt(w2)
    h= np.sqrt(h2)

    ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

    error_sum = np.sum([ellipse_model(x,y) for x,y in data])
    # print('fitting error = %.3f' % (error_sum))

    if np.isnan(cx) or np.isnan(cy) or np.isnan(w) or np.isnan(h) or np.isnan(theta):
        output_file = "bug_log.txt"
        with open(output_file, "a+") as file:
            file.write(f"get some NaN in ellipse module\n")
            file.write(f"{cx} {cy} {w} {h} {theta}\n")
        return 0, 0, 0, 0, 0

    return (cx,cy,w,h,theta)

# 限制两帧之间预测坐标距离
THRESHOLD = 3000 # 3000 即无限制
FIRST = False
pre_cx = 0
pre_cy = 0

def update_position(x, y, a, b, THRESHOLD):
    # 计算两点之间的距离
    distance = math.sqrt((a - x) ** 2 + (b - y) ** 2)
    
    # 如果距离小于阈值，则更新坐标
    if distance < THRESHOLD:
        return a, b
    else:
        # 计算单位向量
        unit_vector_x = (a - x) / distance
        unit_vector_y = (b - y) / distance
        
        # 计算新的坐标，保持距离为 THRESHOLD
        new_x = x + unit_vector_x * THRESHOLD
        new_y = y + unit_vector_y * THRESHOLD
        
        return new_x, new_y
    
x_bias = 0
y_bias = 0

def set_bias(x, y):
    global x_bias, y_bias
    x_bias = x
    y_bias = y

def vision_func(frame, mode=0):
    '''
    接受一帧图像，返回一个预测pose；以及在屏幕上显示识别结果
    pose具体来说是 limbus, flag
    postion = limbus[0]
    direction = limbus[1]
    mode 是留给 眼球中心估计的，暂时不用管
    '''

    global THRESHOLD, FIRST, pre_cx, pre_cy # 维护预测坐标
    # 返回值
    limbus = []
    flag = False

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image_gray,(3,3),0)
    ret,thresh1 = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    image = 255 - closing
    # because of version
    # _,contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hull = []

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))  # 50个点返回一次, w和h要输入直径
                    
    # cnt = sorted(hull, key=cv2.contourArea)
    # maxcnt = cnt[-1]
    for con in hull:
        approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
        area = cv2.contourArea(con)
        if(len(approx) > 10 and area > 250): 
            cx,cy,w,h,theta = fit_rotated_ellipse_ransac(con.reshape(-1,2))

            # 限制两帧之间预测的图像坐标距离  
            if not (cx==0 and cy==0 and w==0 and h==0 and theta==0):
                if not FIRST:
                    FIRST = True
                    pre_cx = cx
                    pre_cy = cy
                else:
                    cx, cy = update_position(pre_cx, pre_cy, cx, cy, THRESHOLD)
                    pre_cx, pre_cy = cx, cy

            # TODO 尝试用realsense得到内参进行输入 
            if mode == 0:
                maj_a = 2 * w
                min_a = 2 * h
                not_sure = theta
                if w < h:
                    maj_a = 2 * h
                    min_a = 2 * w
                    not_sure = theta + 0.5 * np.pi
                limbus, flag = ellipse_to_limbus(cx + x_bias, cy + y_bias, maj_a, min_a, not_sure, True, mode)

            elif mode == 1: # in this mode, arm don't move and wait until points is enough
                limbus, flag = ellipse_to_limbus(cx + x_bias, cy + y_bias, w*2, h*2, theta, True, mode)
                if flag:
                    eyeball_centre = limbus[0]
                    mode = 0 # go back to detect mode
                    print(f"get eyeball centre: {eyeball_centre}")

            # NOTE：你可以在此获得椭圆的信息，中心cx cy; 两轴w h；偏转角theta  
            cv2.ellipse(frame,(int(cx),int(cy)),(int(w),int(h)),theta*180.0/np.pi,0.0,360.0,(0,255,0),1)
            cv2.drawMarker(frame, (int(cx),int(cy)),(0, 0, 255),cv2.MARKER_CROSS,2,1)
            # TODO 输出眼球几何中心，和注视线，需要将三维坐标投影回二维平面，但这个感觉有点难

            break # only one eye
    return limbus,flag