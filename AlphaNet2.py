import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn as nn
import torch
import numpy as np

class AlphaNet2D(nn.Module):
    def __init__(self):
        super(AlphaNet2D, self).__init__()
        self.embedding = nn.Embedding(5000, 5)

        # self.fc1 = nn.Conv2d(1, 16, 1)
        # self.fc2 = nn.Linear(28, 10)

        self.conv1_1 = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.bn1_1 = nn.BatchNorm2d(16)
        self.sigmoid1_1 = torch.nn.ReLU()
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.bn1_2 = nn.BatchNorm2d(16)
        self.sigmoid1_2 = torch.nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.bn2_1 = nn.BatchNorm2d(32)
        self.sigmoid2_1 = torch.nn.ReLU()
        self.conv2_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.bn2_2 = nn.BatchNorm2d(32)
        self.sigmoid2_2 = torch.nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn3_1 = nn.BatchNorm2d(64)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.bn3_2 = nn.BatchNorm2d(64)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        self.fc3 = nn.Linear(960, 1)

    def forward(self, x, code):

        '''x0 = torch.ones((x.shape[0], x.shape[1], x.shape[2], 12)).cuda()
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                for k in range(x0.shape[2]):
                    a = self.embedding(code[i])
                    x0[i, j, k, 0:5] = a[0, 0:5]
                    x0[i, j, k, 5:12] = x[i, j, k, :]'''

        '''x1 = torch.ones((x.shape[0], x.shape[1], x.shape[2], 20)).cuda()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x1[i, j, k] = self.fc1(x[i, j, k])'''

        '''x2 = torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], 10)).cuda()
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                for k in range(x0.shape[2]):
                    x2[i, j, k] = self.fc2(x1[i, j, k])'''

        x2 = self.conv1_1(x)
        x2 = self.bn1_1(x2)
        x2 = self.sigmoid1_1(x2)
        x2 = self.conv1_2(x2)
        x2 = self.bn1_2(x2)
        x2 = self.sigmoid1_2(x2)
        x2 = self.maxpool1(x2)

        x2 = self.conv2_1(x2)
        x2 = self.bn2_1(x2)
        x2 = self.sigmoid2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.sigmoid2_2(x2)
        x2 = self.maxpool2(x2)

        x2 = self.conv3_1(x2)
        x2 = self.bn3_1(x2)
        x2 = self.relu3_1(x2)

        x2 = self.conv3_2(x2)
        x2 = self.bn3_2(x2)
        x2 = self.relu3_2(x2)
        # x2 = self.maxpool3(x2)
        # print(x2.shape)

        out = x2.reshape(x2.shape[0], 960)
        out = self.fc3(out)

        return out

class AlphaNet1D(nn.Module):
    def __init__(self):
        super(AlphaNet1D, self).__init__()
        # self.embedding = nn.Embedding(5000, 5)

        # self.fc1 = nn.Conv1d(1, 16, 1)
        # self.fc2 = nn.Linear(28, 10)

        self.conv1_1 = nn.Conv1d(12, 16, 3, 1, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.activate1_1 = torch.nn.ReLU()
        self.conv1_2 = nn.Conv1d(16, 16, 3, 1, padding=1)
        self.bn1_2 = nn.BatchNorm1d(16)
        self.activate1_2 = torch.nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2_1 = nn.Conv1d(16, 32, 3, 1, padding=1)
        self.bn2_1 = nn.BatchNorm1d(32)
        self.activate2_1 = torch.nn.ReLU()
        self.conv2_2 = nn.Conv1d(32, 32, 3, 1, padding=1)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.activate2_2 = torch.nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv3_1 = nn.Conv1d(32, 64, 3, 1, padding=1)
        self.bn3_1 = nn.BatchNorm1d(64)
        self.activate3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.activate3_2 = nn.BatchNorm1d(64)
        self.relu3_2 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(2)

        self.fc3 = nn.Linear(320, 2)

    def forward(self, x, code):

        '''x0 = torch.ones((x.shape[0], x.shape[1], x.shape[2], 12)).cuda()
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                for k in range(x0.shape[2]):
                    a = self.embedding(code[i])
                    x0[i, j, k, 0:5] = a[0, 0:5]
                    x0[i, j, k, 5:12] = x[i, j, k, :]'''

        '''x1 = torch.ones((x.shape[0], x.shape[1], x.shape[2], 20)).cuda()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    x1[i, j, k] = self.fc1(x[i, j, k])'''

        '''x2 = torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], 10)).cuda()
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                for k in range(x0.shape[2]):
                    x2[i, j, k] = self.fc2(x1[i, j, k])'''

        # for i in range(64):
        #     print(x[i])
        # print(x)
        x2 = self.conv1_1(x)
        # print(x2[0], x2[1])
        # x2 = self.bn1_1(x2)
        x2 = self.activate1_1(x2)
        x2 = self.conv1_2(x2)
        # x2 = self.bn1_2(x2)
        x2 = self.activate1_2(x2)
        x2 = self.maxpool1(x2)

        x2 = self.conv2_1(x2)
        # x2 = self.bn2_1(x2)
        x2 = self.activate2_1(x2)
        x2 = self.conv2_2(x2)
        # x2 = self.bn2_2(x2)
        x2 = self.activate2_2(x2)
        x2 = self.maxpool2(x2)

        x2 = self.conv3_1(x2)
        # x2 = self.bn3_1(x2)
        x2 = self.activate3_1(x2)
        x2 = self.conv3_2(x2)
        # x2 = self.bn3_2(x2)
        x2 = self.activate3_2(x2)
        # print(x2[0], x2[1])
        # x2 = self.maxpool3(x2)
        # print(x2.shape)

        out = x2.reshape(x2.shape[0], 320)
        out = self.fc3(out)

        return out

class AlphaNetFC(nn.Module):
    def __init__(self):
        super(AlphaNetFC, self).__init__()
        # self.embedding = nn.Embedding(5000, 5)

        self.fc1 = nn.Linear(240, 28)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(28, 10)
        self.activate2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x, code):

        x = x.reshape(x.shape[0], 240)
        x = self.fc1(x)
        x = self.activate1(x)
        x = self.fc2(x)
        x = self.activate2(x)
        out = self.fc3(x)

        return out


if __name__ == '__main__':
    b = torch.ones((128, 1, 20, 7)).cuda()
    # b = b.cuda()
    code = np.ones((128, 1)) * 1000
    code = torch.LongTensor(code).cuda()
    # code = code.cuda()
    net = AlphaNet2D().cuda()
    # net = net.cuda()
    print(net(b, code))
