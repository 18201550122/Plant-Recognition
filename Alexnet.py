import torch
import torchvision
import torchvision.models
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from PIL import ImageFile
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((120, 120)), #这种预处理的地方尽量别修改，修改意味着需要修改网络结构的参数，如果新手的话请勿修改
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
# dataPath=r'D:\plant_recognition\fit_model_Adaptive_clipping-trans_extended_test'
dataPath=r'D:\plant_recognition\TEST'
'''                       可调代码点    加载数据            ******************************************************************************               '''
train_data = torchvision.datasets.ImageFolder(root=dataPath+"\\train", transform=data_transform["train"])

traindata = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=0)

# test_data = torchvision.datasets.CIFAR10(root = "./data" , train = False ,download = False,
#                                           transform = trans)
test_data = torchvision.datasets.ImageFolder(root=dataPath+"\\val", transform=data_transform["val"])

train_size = len(train_data)  # 求出训练集的长度
test_size = len(test_data)  # 求出测试集的长度
print(train_size)  # 输出训练集的长度
print(test_size)  # 输出测试集的长度
testdata = DataLoader(dataset=test_data, batch_size=32, shuffle=True,
                      num_workers=0)  # windows系统下，num_workers设置为0，linux系统下可以设置多进程

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 120, 120]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 46),  # 自己的数据集是几种，这个7就设置为几
            '''                       可调代码点        自己的数据集是几种，这个46就设置为几        ******************************************************************************               '''
        )

    def forward(self, x):
        x = self.model(x)
        return x

alexnet1 = alexnet()
print(alexnet1)
alexnet1.to(device)  # 将模型放入GPU
test1 = torch.ones(64, 3, 120, 120)  # 输入数据测试一下模型能不能跑

test1 = alexnet1(test1.to(device))
print(test1.shape)

'''                       可调代码点                ******************************************************************************               '''
epoch = 2  # 这里是训练的轮数
learning = 0.0001  # 学习率
optimizer = torch.optim.Adam(alexnet1.parameters(), lr=learning)  # 优化器
loss = nn.CrossEntropyLoss()  # 损失函数
'''                       可调代码点                ******************************************************************************               '''

train_loss_all = []
train_accur_all = []
test_loss_all = []
test_accur_all = []
for i in range(epoch):
    train_loss = 0
    train_num = 0.0
    train_accuracy = 0.0
    alexnet1.train()
    train_bar = tqdm(traindata)
    for step, data in enumerate(train_bar):
        img, target = data
        optimizer.zero_grad()
        outputs = alexnet1(img.to(device))

        loss1 = loss(outputs, target.to(device))
        outputs = torch.argmax(outputs, 1)
        loss1.backward()
        optimizer.step()
        train_loss += abs(loss1.item()) * img.size(0)
        accuracy = torch.sum(outputs == target.to(device))
        train_accuracy = train_accuracy + accuracy
        train_num += img.size(0)

    print("epoch：{} ， train-Loss：{} , train-accuracy：{}".format(i + 1, train_loss / train_num,
                                                                train_accuracy / train_num))
    train_loss_all.append(train_loss / train_num)
    train_accur_all.append(train_accuracy.double().item() / train_num)

    test_loss = 0
    test_accuracy = 0.0
    test_num = 0
    alexnet1.eval()
    with torch.no_grad():
        test_bar = tqdm(testdata)
        for data in test_bar:
            img, target = data

            outputs = alexnet1(img.to(device))

            loss2 = loss(outputs, target.to(device))
            outputs = torch.argmax(outputs, 1)
            test_loss = test_loss + abs(loss2.item()) * img.size(0)
            accuracy = torch.sum(outputs == target.to(device))
            test_accuracy = test_accuracy + accuracy
            test_num += img.size(0)

    print("test-Loss：{} , test-accuracy：{}".format(test_loss / test_num, test_accuracy / test_num))
    test_loss_all.append(test_loss / test_num)
    test_accur_all.append(test_accuracy.double().item() / test_num)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_loss_all,
         "ro-", label="Train loss")
plt.plot(range(epoch), test_loss_all,
         "bs-", label="test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_accur_all,
         "ro-", label="Train accur")
plt.plot(range(epoch), test_accur_all,
         "bs-", label="test accur")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()

import random
r1 = random.randint(1,1000)      #返回 [a,b] 之间的任意整数
import time
now = time.localtime()
nowt = time.strftime("%Y-%m-%d-%H_%M_%S--", now)  #对时间进行格式化
# print(nowt)
plt.savefig('Alexnet'+nowt+str(r1)+'.png')
'''                       可调代码点       结果保存         ******************************************************************************               '''

plt.show()

torch.save(alexnet1.state_dict(), "Alexnet"+nowt+str(r1)+".pth")
'''                       可调代码点       模型保存         ******************************************************************************               '''
print("模型已保存")

