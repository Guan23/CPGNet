import numpy as np
import torch
from torch import nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 这里的MLP我用的1x1Conv2d来实现的，当然也可以用其他网络层
        self.conv1 = nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(1, 1))

    def forward(self, bev_feature, rv_feature, point_feature):
        # 融合模块，作者说就一个concat再加两个MLP，比较简单，三个输入分别来自bev，rv，point三个分支
        x = torch.cat((bev_feature, rv_feature, point_feature), dim=1)
        x = self.conv1(x)
        output = self.conv2(x)
        return output


if __name__ == "__main__":
    print("\n", "-" * 50, " start ", "-" * 50, "\n")

    model = MyNet()
    bev_feature = torch.ones((1, 64, 125655, 1))
    rv_feature = torch.ones((1, 64, 125655, 1))
    point_feature = torch.ones((1, 64, 125655, 1))
    output = model(bev_feature, rv_feature, point_feature)
    print(output.shape)  # torch.Size([1, 96, 125655, 1])

    print("\n", "-" * 51, " end ", "-" * 51, "\n")
