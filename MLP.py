import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 如果按照PointNet的源码，则输入为(b, in_c, h, w)，输出为(b, out_c, h, 1)
        # (1, 1, 120000, 9) -> (1, 64, 120000, 1)
        # 其中b为batchsize，c为channel，这两维都是扩充出来的，h为点云数量n，w为点云的属性，比如4(x, y, z, i)
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 4))

    def forward(self, input):
        output = self.conv(input)
        return output


if __name__ == "__main__":
    print("\n", "-" * 50, " start ", "-" * 50, "\n")

    model = MLP()

    lidar = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [5, 3, 4, 5],
        ], dtype=torch.float32
    )
    in_tensor = lidar.unsqueeze(0).unsqueeze(1)  # 扩充前两维，b和c，实际中只需要扩充第二维c即可

    print(in_tensor)
    print(in_tensor.shape)
    out = model(in_tensor)
    print("*" * 50)
    print(out)
    print(out.shape)

    print("\n", "-" * 51, " end ", "-" * 51, "\n")
