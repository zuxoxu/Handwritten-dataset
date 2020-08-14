import torch
from PIL import Image
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from torch.nn import functional as F  #调用F.函数
import glob
import os

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

# 加在网络和参数
model = ConvNet()
model.load_state_dict(torch.load('model.ckpt'))
model.eval()

file_list = glob.glob(os.path.join('./number1/', '*'))
grid_rows = len(file_list) / 5 + 1

for i, file in enumerate(file_list):
    image = Image.open(file).resize((28, 28))
    gray_image = image.convert('L')
    transform = transforms.ToTensor()
    im_data = transform(gray_image)
    im_data = im_data.resize(1, 1, 28, 28)
    outputs = model(im_data)
    _, pred = torch.max(outputs, 1)
    plt.subplot(grid_rows, 5, i + 1)
    plt.imshow(gray_image)
    plt.title("is {}".format(pred.item()), fontsize=24)
    plt.axis('off')
    print('[{}]预测数字为: [{}]'.format(file, pred.item()))
plt.show()
