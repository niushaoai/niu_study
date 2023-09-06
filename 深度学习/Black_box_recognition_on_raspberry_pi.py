import cv2
import numpy as np
import serial
import time
#open serial
ser = serial.Serial("/dev/ttyAMA0", 115200)#set up serial
# 初始化摄像头
 
cap = cv2.VideoCapture(0,cv2.CAP_GSTREAMER)

# 设置摄像头分辨率
cap.set(3, 500)
cap.set(4, 500)

# 定义红色阈值
red_lower = np.array([0, 100, 100])
red_upper = np.array([30, 255, 255])

# 循环读取视频帧
while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 裁剪图像
#    cropped_image = frame[100:640, 100:480]

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 轮廓查找
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大轮廓和最大面积
    max_contour = None
    max_area = 0
    zuobiao = [0,0,0,0,0,0,0,0]
    charzuob=""
    hong = [0,0]
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 如果面积大于阈值，则更新最大轮廓和最大面积
        if area > max_area:
            max_area = area
            max_contour = contour

    # 如果找到最大轮廓
    if max_contour is not None:
        # 计算最大轮廓的顶点坐标
        epsilon = 0.01 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            # 获取黑色方框的四点坐标
            x1, y1 = approx[0][0]
            x2, y2 = approx[1][0]
            x3, y3 = approx[2][0]
            x4, y4 = approx[3][0]
            zuobiao[0] = x1
            zuobiao[1] = y1
            zuobiao[2] = x2
            zuobiao[3] = y2
            zuobiao[4] = x3
            zuobiao[5] = y3
            zuobiao[6] = x4
            zuobiao[7] = y4
        for j in range(8):
            if zuobiao[j]<10:
                temchar = "00%d"%zuobiao[j]
            elif zuobiao[j]<100:
                temchar = "0%d"%zuobiao[j]
            else :
                temchar = str(zuobiao[j])
            #print(temchar)
            charzuob=charzuob+temchar
        charzuob= charzuob+'A'
        if len(charzuob) == 25:
            ser.write(charzuob  .encode('utf-8'))                 
        # 绘制轮廓
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # 绘制顶点
#        for point in approx
#        	cv2.circle(cropped_image, tuple(point[0]), 5, (0, 0, 255), -1)

        # 计算顶点之间的距离和角度信息
        distances = []
        angles = []
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i+1) % len(approx)][0]
            distance = np.linalg.norm(p2 - p1)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
            distances.append(distance)
            angles.append(angle)

        # 在图像上显示距离和角度信息
#        for i in range(len(distances)):
#            cv2.putText(cropped_image, "Distance: {:.2f}".format(distances[i]), (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#            cv2.putText(cropped_image, "Angle: {:.2f}".format(angles[i]), (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 根据红色阈值提取红色激光区域的掩码
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    # 将掩码应用到原始图像上，获得红色激光的结果图像
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)
    # 找到红色激光区域的轮廓
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓并获取红色激光图像的坐标
    for contour_red in contours_red:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour_red)
        
        # 在图像上绘制边界框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 打印红色激光图像的坐标
        print(x,y)

    # 显示图像
    cv2.imshow("Image", np.hstack([frame, red_result]))

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()

# 清除窗口
cv2.destroyAllWindows()
