import numpy as np
import cv2 as cv
import random
# 设置putText函数字体
font = cv.FONT_HERSHEY_SIMPLEX
import os

def check_persent_of_white(img):
    # 计算白色像素点的个数
    count = 0
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 250:
                count += 1
    # 计算白色像素点所占比例
    persent = count / (img.shape[0] * img.shape[1])
    return persent

def find_squares(img, target):
    squares = []
    # 图像预处理
    img = cv.resize(img, (324, 324), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    raw_img = img.copy()
    # 加入白色边框
    img = cv.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 255), 30)
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    ret, binary = cv.threshold(gray, 245, 255, cv.THRESH_TOZERO)
    # 边缘检测
    bin = cv.Canny(binary, 4, 4, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    bin = cv.dilate(bin, kernel, iterations=2)
    # bin = cv.erode(bin, kernel, iterations=2)
    bin = cv.bitwise_not(bin)
    contours, _hierarchy = cv.findContours(bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    index = 0
    cnt_list = []
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        #
        temp = cv.drawContours(img, contours, -1, (0, 0, 255), 1)
        # 把多边形拟合出矩形
        x, y, w, h = cv.boundingRect(cnt)
        if w > 50 and h > 50:
            cnt_list.append((x, y, w, h))
    # 检查是否互相包含
    for i in range(len(cnt_list)):
        for j in range(len(cnt_list)):
            if i != j:
                # 如果x1>x2 and y1>y2 and x1+w1<x2+w2 and y1+h1<y2+h2
                if cnt_list[i][0] <= cnt_list[j][0] and \
                    cnt_list[i][1] <= cnt_list[j][1] and \
                    cnt_list[i][0] + cnt_list[i][2] >= cnt_list[j][0] + cnt_list[j][2] and \
                    cnt_list[i][1] + cnt_list[i][3] >= cnt_list[j][1] + cnt_list[j][3]:
                    cnt_list[i] = (0, 0, 0, 0)
    save_num = 0
    for x, y, w, h in cnt_list:
        if w > 50 and h > 50:
            cv.rectangle(img, (x, y), (x + w, y + h), (random.randint(0,250), random.randint(0,250), 0), 2)
            cnt = cnt.reshape(-1, 2)
            index = index + 1
            # cv.putText(ori_img, ("#%d" % index), (cx, cy), font, 0.7, (255, 0, 255), 2)
            squares.append(cnt)
            # 保存图片
            per = check_persent_of_white(raw_img[y:y + h, x:x + w])
            if per < 0.51:
                # print(f"保存图片{target}_{index}.jpg")
                flag = cv.imwrite(f"{target}_{index}.jpg", raw_img[y:y + h, x:x + w])
                save_num += 1
                if not flag:
                    raise Exception("保存图片失败")
    if save_num == 0:
        print("未保存图片")
        cv.imwrite(f"{target}_{index}.jpg", raw_img)
        print(f"保存图片{target}_0000.jpg")


def main():
    ROOT_PATH = "ip102_v1.1/test_proceed/"
    # 遍历文件夹
    for root, dirs, files in os.walk(ROOT_PATH):
        # print(root,dir,files)
        for file in files:
            if file.endswith(".jpg"):
                src = os.path.join(root, file)
                img = cv.imread(src)
                name = os.path.splitext(src)[0]
                find_squares(img, name)
                os.remove(src)
        print(f"{root}文件夹处理完成")

if __name__ == '__main__':
    main()

