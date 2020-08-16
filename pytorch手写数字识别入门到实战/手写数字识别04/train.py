#摄像头识别手写数字
import numpy as np
import torch
import cv2 as cv
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

cap = cv.VideoCapture(0)      #打开摄像头，0为默认笔记本自带摄像头
while (cap.isOpened()):      #当摄像头开启时运行下面代码
    ret, frame = cap.read()       #第一个参数ret为True或者False,代表有没有读取到图片第二个参数frame表示截取到一帧的图片
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)  #将获得的每一帧转化成灰色图
    imgG = cv.GaussianBlur(gray, (5, 5), 0)      #进行高斯模糊
    erosion = cv.erode(imgG, (3, 3), iterations=3)    #再对图像进行腐蚀操作
    dilate = cv.dilate(erosion, (3, 3), iterations=3)  #再膨胀
    edged = cv.Canny(dilate, 80, 200, 255)   #边缘检测
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  #轮廓发现 1参数图像源
                                                                        #2参数检测方式 3参数用尽可能少的像素点表示轮廓
    digitcnts = []    #创建空数组
    for i in contours:
        (x, y, w, h) = cv.boundingRect(i)  #获得图像的矩形框数据
        if w < 100 and h > 45 and h < 160:
            digitcnts.append(i)
    m = 0
    for c in digitcnts:
        (x, y, w, h) = cv.boundingRect(c)
        m += 1
        roi1 = frame[y:y + h, x:x + w]
        height, width, channel = roi1.shape   #获得矩形框的长宽和通道数
        for i in range(height):
            for j in range(width):
                b, g, r = roi1[i, j]
                if g > 180:
                    b = 255
                    r = 255
                    g = 255
                else:
                    b = 0
                    g = 0
                    r = 0
                roi1[i, j] = [b, g, r]
        roi1 = 255 - roi1
        roi2 = cv.copyMakeBorder(roi1, 30, 30, 30, 30, cv.BORDER_CONSTANT, value=[0, 0, 0])
        cv.imwrite('%d.png' % m, roi2)
        img1 = cv.imread('%d.png' % m,0)
        img1 = cv.GaussianBlur(img1, (5, 5), 0)
        #img1 = cv.erode(img1, (3, 3), iterations=3)
        img1 = cv.dilate(img1, (3, 3), iterations=3)
        #cv.imwrite('%d_111.png' % m, img1)
        img2 = cv.resize(img1, (28, 28), interpolation=cv.INTER_CUBIC)
        img3 = np.array(img2) / 255
        img4 = np.reshape(img3, [-1, 784])

        images = torch.tensor(img4, dtype=torch.float32)
        images = images.resize(1, 1, 28, 28)

        model = ConvNet()
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()
        outputs = model(images)
        values, indices = outputs.data.max(1)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 0)
        cv.putText(frame, str(indices[0]), (x, y), font, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("capture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
          break
cap.release()
cv.destroyAllWindows()