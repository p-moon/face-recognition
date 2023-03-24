import cv2
import os

"""
加载训练好的模型来识别人脸
"""

# 加载级联分类器
face_cascade = cv2.CascadeClassifier('conf/haarcascade_frontalface_default.xml')

# 加载人脸识别模型
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("data"):
    recognizer.read('data/trainer.yml')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 遍历每个人脸并识别
    for (x, y, w, h) in faces:
        # 识别人脸
        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 100:
            name = "Person {}-{}".format(label, confidence)
        else:
            name = "Unknown"

        # 在图像上标记人脸和姓名
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('frame', frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
