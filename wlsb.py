import cv2
import numpy as np

def get_center_point(image):
    image = cv2.imread('./data/001.jpg')
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200)
    center_point_x, center_point_y = 0, 0
    if circles is not None:    
        circles = np.round(circles[0, :]).astype('int')
        for (x, y, r) in circles:
            center_point_x = x
            center_point_y = y
    return center_point_x, center_point_y

def get_init_point(image):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX) 
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX=(255*(gradX-minVal)/(maxVal-minVal))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 124, 250,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs_ref2, locs_ref3 = [], []
    for (i, c_ref) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c_ref)
        if 402>w and 350<w:
            locs_ref2.append((x, y, w, h))
        if 36>w and 13<w and 34>h and 18<h:
            locs_ref3.append((x, y, w, h))
    locs_ref3 = sorted(locs_ref3, key=lambda x: x[0])
    zero_point_x = locs_ref3[0][0]
    zero_point_y = locs_ref3[0][1] + locs_ref3[0][3]
    return locs_ref2, zero_point_x, zero_point_y

def get_parallel_center_point(locs_ref2, center_point_y):
    dis_x = locs_ref2[0][0]
    dis_y = locs_ref2[0][1]
    dis_w = locs_ref2[0][2]
    dis_h = locs_ref2[0][3]
    left_point_x = dis_x
    left_point_y = center_point_y
    right_point_x = dis_x + dis_w
    right_point_y = center_point_y
    return left_point_x, left_point_y, right_point_x, right_point_y

def get_ratio():
    ratio = (1.3-0.18)/math.pi
    return ratio

def get_point_with_huofu(gray)
    img = cv2.GaussianBlur(gray,(3,3),0)
    result=img.copy()
    cannyImage = cv2.Canny(img,120,243.20999999999998,apertureSize = 3)    
    HoughLines=cv2.HoughLines(cannyImage,1, np.pi/ 180, 40 + 1)  
    for line in HoughLines[0]:  
        rho = line[0]
        theta = line[1]
        if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
            pt1 = (int(rho/np.cos(theta)),0)
            pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0]) 
            cv2.line( result, pt1, pt2, (255))
        else:
            pt1 = (0,int(rho/np.sin(theta))) 
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            cv2.line(result,pt1,pt2,(255),1)

    AB = [center_point_x,center_point_y,zero_point_x,zero_point_y]
    CD = [pt2[0],pt2[1],pt1[0],pt1[1]]
    return AB, CD

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def show_image(image, AB, CD, ratio, center_point_x, center_point_y):
    degree = angle(AB, CD)  
    scale_value = degree * ratio　
    cv2.putText(image, str(scale_value), (center_point_x-40,center_point_y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("image",image)
    cv2.waitKey(0)
