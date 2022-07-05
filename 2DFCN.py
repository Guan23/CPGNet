# 2022-05-12 09:52
# created by guan
# 根据CPGNet的论文，把他的2D FCN语义提取模块复现了一下，并测试了时间，
# 对于rv = torch.ones(1, 64, 64, 2048)，平均每帧 11 ms(论文中在2080Ti上的总infer时间是 43 ms，差不多)
# 对于bev = torch.ones(1, 64, 600, 600)，时间未测试
# trainable_params的数量为940576

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

a = 0.8


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        out = self.sigmoid(x)
        max_out = torch.max(out, dim=1, keepdim=True).values
        b, c, h, w = out.shape
        # assert一下max_out的尺寸是否为(b, 1, h, w)
        return max_out


# [b, c, h, w] -> [b, 2c, h/2, w/2]
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        # self.act_front = nn.LeakyReLU()

        # 双降采样块，即3*3使用stride降采样，而1*1加一个maxpooling降采样
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1)
        # self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2)

        self.act_back = nn.ReLU()

    def forward(self, x):
        # shortcut = self.conv(x)
        # shortcut = self.act_front(shortcut)

        resA = self.conv3x3(x)
        # resA = self.act1(resA)
        resA = self.bn1(resA)

        resB = self.conv1x1(x)
        # resB = self.act2(resB)
        resB = self.bn2(resB)
        resB = self.maxpool(resB)

        output = resA + resB
        output = self.act_back(output)
        return output


# x1:(2c, h/2, w/2), x2:(c, h, w) -> (c, h, w)
class UpBlock(nn.Module):
    def __init__(self, c1_in_channels, c1_out_channels, c2_in_channels, c2_out_channels):
        super(UpBlock, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels

        self.conv1 = nn.Conv2d(c1_in_channels // 4, c1_out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1_out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(c2_in_channels, c2_out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c2_out_channels)
        self.act2 = nn.ReLU()

        self.attention = Attention(c1_out_channels, c2_out_channels)

    # x1:(2c, h/2, w/2), x2:(c, h, w)
    def forward(self, x1, x2):
        x1 = nn.PixelShuffle(2)(x1)  # up = 2, upsample, (2c, h/2, w/2) -> (2c/(2^2)=c/2, h, w)
        up1 = self.conv1(x1)
        up1 = self.bn1(up1)
        up1 = self.act1(up1)

        up2 = self.conv2(x2)
        up2 = self.bn2(up2)
        up2 = self.act2(up2)

        up = self.attention(up1, up2)
        output = ((a * up) * up2) + (((1 - a) * up) * up1)
        return output


class CPGNet(nn.Module):
    def __init__(self, nclasses):
        super(CPGNet, self).__init__()
        self.nclasses = nclasses

        self.resBlock1 = ResBlock(in_channels=64, out_channels=32)
        self.resBlock2 = ResBlock(in_channels=32, out_channels=64)
        self.resBlock3 = ResBlock(in_channels=64, out_channels=128)
        self.resBlock4 = ResBlock(in_channels=128, out_channels=128)
        # self.resBlock4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )
        self.upBlock1 = UpBlock(c1_in_channels=128, c1_out_channels=96, c2_in_channels=128, c2_out_channels=96)
        self.upBlock2 = UpBlock(c1_in_channels=96, c1_out_channels=64, c2_in_channels=64, c2_out_channels=64)
        self.upBlock3 = UpBlock(c1_in_channels=64, c1_out_channels=64, c2_in_channels=32, c2_out_channels=64)
        self.upBlock4 = UpBlock(c1_in_channels=64, c1_out_channels=64, c2_in_channels=64, c2_out_channels=64)

    def forward(self, input):
        d1 = self.resBlock1(input)  # [1, 64, 64, 2048] -> [1, 32, 32, 1024]
        d2 = self.resBlock2(d1)  # [1, 32, 32, 1024] -> [1, 64, 16, 512]
        d3 = self.resBlock3(d2)  # [1, 64, 16, 512] -> [1, 128, 8, 256]
        d4 = self.resBlock4(d3)  # [1, 128, 8, 256] -> [1, 128, 8, 256]

        u1 = self.upBlock1(d4, d3)  # [1, 96, 8, 256]
        u2 = self.upBlock2(u1, d2)  # [1, 64, 16, 512]
        u3 = self.upBlock3(u2, d1)  # [1, 64, 32, 1024]
        output = self.upBlock4(u3, input)  # [1, 64, 64, 2048]

        return output


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    print("\n---------------- start ----------------\n")
    device = 'cuda'
    rv = torch.ones(1, 64, 64, 2048)
    bev = torch.ones(1, 64, 600, 600)
    mymodel = CPGNet(20)
    mymodel.to(device)
    rv = rv.to(device)
    # bev = bev.to(device)

    # out = mymodel(bev)

    params_dict = get_parameter_number(mymodel)
    print(params_dict['Total'])
    print(params_dict['Trainable'])

    infer = []

    for i in range(150):
        t0 = time.time()
        out1 = mymodel(rv)
        t1 = time.time()
        infer.append((t1 - t0))
        print(f"the {i+1} frame infered!")

    del (infer[0])
    print((np.mean(infer)))

    # writer = SummaryWriter("../doc")
    # writer.add_graph(model=mymodel, input_to_model=rv)
    # writer.close()

    print("\n----------------- end -----------------\n")
