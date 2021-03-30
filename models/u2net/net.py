import torch
import torch.nn as nn

from models.layers import gap2d
from models.u2net_layers import RSU7, RSU4, RSU5, RSU6, RSU4F


class Net(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, num_classes=20, *args, **kwargs):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

        self.classifier1 = nn.Conv2d(64, self.num_classes, 1, bias=False)
        self.classifier2 = nn.Conv2d(128, self.num_classes, 1, bias=False)
        self.classifier3 = nn.Conv2d(256, self.num_classes, 1, bias=False)
        self.classifier4 = nn.Conv2d(512, self.num_classes, 1, bias=False)
        self.classifier5 = nn.Conv2d(512, self.num_classes, 1, bias=False)
        self.classifier6 = nn.Conv2d(512, self.num_classes, 1, bias=False)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)

        hx1_gap = gap2d(hx1, keepdims=True)
        hx2_gap = gap2d(hx2, keepdims=True)
        hx3_gap = gap2d(hx3, keepdims=True)
        hx4_gap = gap2d(hx4, keepdims=True)
        hx5_gap = gap2d(hx5, keepdims=True)
        hx6_gap = gap2d(hx6, keepdims=True)

        cls_lab1 = self.classifier1(hx1_gap).view(-1, self.num_classes)
        cls_lab2 = self.classifier2(hx2_gap).view(-1, self.num_classes)
        cls_lab3 = self.classifier3(hx3_gap).view(-1, self.num_classes)
        cls_lab4 = self.classifier4(hx4_gap).view(-1, self.num_classes)
        cls_lab5 = self.classifier5(hx5_gap).view(-1, self.num_classes)
        cls_lab6 = self.classifier6(hx6_gap).view(-1, self.num_classes)
        cls_lab0 = cls_lab1 + cls_lab2 + cls_lab3 + cls_lab4 + cls_lab5 + cls_lab6

        return cls_lab0, cls_lab1, cls_lab2, cls_lab3, cls_lab4, cls_lab5, cls_lab6


if __name__ == '__main__':
    from torchsummary import summary

    model = Net()

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    y0, y1, y2, y3, y4, y5, y6 = y

    assert y0.shape == (2, 20)
    assert y1.shape == (2, 20)
    assert y2.shape == (2, 20)
    assert y3.shape == (2, 20)
    assert y4.shape == (2, 20)
    assert y5.shape == (2, 20)
    assert y6.shape == (2, 20)
