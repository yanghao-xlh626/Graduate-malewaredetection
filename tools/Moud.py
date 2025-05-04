import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class ResidualBlock(nn.Module):
    # 这是一个残差块,包含3个卷积层(两个1x1,一个3x3)和一个shortcut分支,用于处理输入和输出尺寸不一致的情况.
    # NOTE:负责将一个[batch,in_channel,height,width]的张量转换为[batch,mid_channel*4,height/2,width/2]
    def __init__(self, in_channel,mid_channel,stride=2):
        self.expansion = 4
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,mid_channel,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)

        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride,padding=1, bias=False)        
        self.bn2 = nn.BatchNorm2d(mid_channel)

        self.conv3 = nn.Conv2d(mid_channel,mid_channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channel*self.expansion)

        # NOTE:此处根据数据量的大小选择rate,小数据量0.1~0.3(太大过拟合),大数量可以0.3~0.5
        self.dropout = nn.Dropout(p=0.5)
        
        self.side_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_channel*self.expansion)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        side_branch_x = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(self.bn3(self.conv3(out)))
        side_branch_out = self.side_conv(side_branch_x)
        out += side_branch_out
        output = self.relu(out)
        # output = self.dropout(self.relu(out))  #NOTE
        return output

class MalexByteplot(nn.Module):
    def __init__(self):
        super(MalexByteplot, self).__init__()
        self.conv_7_64_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 16, stride=1),
            ResidualBlock(64, 32, stride=1),
            ResidualBlock(128, 32, stride=1),
            ResidualBlock(128, 64, stride=2),
            ResidualBlock(256, 64, stride=1),
            ResidualBlock(256, 128, stride=2),
            ResidualBlock(512, 128, stride=1),
            ResidualBlock(512, 256, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
    def forward(self,x):
        # 返回值的尺寸为[batch,1024,1,1]
        x = self.conv_7_64_2(x)
        x =self.residual_blocks(x)
        x = self.avgpool(x)
        return x

class MalexBigram(nn.Module):
    def __init__(self):
        super(MalexBigram, self).__init__()
        self.conv_7_64_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 16, stride=1),
            ResidualBlock(64, 32, stride=1),
            ResidualBlock(128, 32, stride=1),
            ResidualBlock(128, 64, stride=2),
            ResidualBlock(256, 64, stride=1),
            ResidualBlock(256, 128, stride=2),
            ResidualBlock(512, 128, stride=1),
            ResidualBlock(512, 256, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,x):
        # 返回值的尺寸为[batch,1024,1,1]
        x = self.conv_7_64_2(x)
        x =self.residual_blocks(x)
        x = self.avgpool(x)
        return x

class MalexResnet(nn.Module):
    def __init__(self, ):  # num_classes 是类别数,默认为2
        super(MalexResnet, self).__init__()
        self.bigram = MalexBigram()
        self.byteplot = MalexByteplot()
        self.fc1 = nn.Linear(1024*10, 1024)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 =nn.Linear(1024,1)


    def forward(self,byte,bigram):
        byte = self.byteplot(byte)
        byte_flat = byte.view(byte.size(0),-1)
        bigram = self.bigram(bigram)
        bigram_flat = bigram.view(bigram.size(0),-1)
        combined = torch.cat((byte_flat,bigram_flat),dim=1)
        output = self.fc2(self.fc1(combined)).squeeze(-1)
        return output
        

# if __name__ == "__main__":
    # x1 = torch.randn(2, 1, 256, 256)  # batch=2, channel=1, size=256*256
    # x2 = torch.randn(2,1,256,256)
    # block = MalexResnet()
    # out = block(x1,x2)
    # print("输出尺寸:", out.shape)
    


        

