import cv2
import numpy as np
import os
import argparse

"""
采集某个用户的人脸信息，并且生成模型写入到data/trainer.yml里面
"""

parser = argparse.ArgumentParser()
parser.add_argument('-un', '--user-id', help='请指定采集的用户id', required=True)
args = parser.parse_args()

user_id = int(args.user_id)

# 创建LBPH人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("data/trainer.yml"):
    recognizer.read('data/trainer.yml')

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('conf/haarcascade_frontalface_default.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

# 定义人脸图像和标签列表
faces = []
labels = []

# 循环读取摄像头数据，直到收集到足够的人脸图像
count = 0
while count < 50:
    # 读取摄像头数据
    ret, frame = cap.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 将每个检测到的人脸图像和标签添加到列表中
    for (x, y, w, h) in faces_rects:
        roi_gray = gray[y:y + h, x:x + w]
        faces.append(roi_gray)
        labels.append(user_id)  # 这里假设所有的图像都是同一个人的，标签为1
        # 显示当前收集到的人脸图像数量
        count += 1
        print("Collected {} faces.".format(count))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 框出人脸
        cv2.imshow('face_mark', frame)
        # 显示当前收集到的人脸图像
        cv2.imshow('face', roi_gray)
        cv2.waitKey(100)

# 训练人脸识别模型
recognizer.train(faces, np.array(labels))

# 保存模型到文件
recognizer.save('data/trainer.yml')

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
