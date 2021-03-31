import torch
import torch.nn.functional as F

from models.u2netp.net import Net


class CAM(Net):

    def __init__(self, *args, **kwargs):
        super(CAM, self).__init__(*args, **kwargs)

    def forward(self, x):
        return self.forward_cam(x)

    def forward_cam(self, x):
        with torch.set_grad_enabled(False):
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

            if self.share_classifier:
                hx6 = F.relu(F.conv2d(hx6, self.classifier.weight))
                hx5 = F.relu(F.conv2d(hx5, self.classifier.weight))
                hx4 = F.relu(F.conv2d(hx4, self.classifier.weight))
                hx3 = F.relu(F.conv2d(hx3, self.classifier.weight))
                hx2 = F.relu(F.conv2d(hx2, self.classifier.weight))
                hx1 = F.relu(F.conv2d(hx1, self.classifier.weight))
            else:
                hx6 = F.relu(F.conv2d(hx6, self.classifier6.weight))
                hx5 = F.relu(F.conv2d(hx5, self.classifier5.weight))
                hx4 = F.relu(F.conv2d(hx4, self.classifier4.weight))
                hx3 = F.relu(F.conv2d(hx3, self.classifier3.weight))
                hx2 = F.relu(F.conv2d(hx2, self.classifier2.weight))
                hx1 = F.relu(F.conv2d(hx1, self.classifier1.weight))
            x_size = x.shape[2:]
            hx1 = hx1[0] + hx1[1].flip(-1)
            hx2 = hx2[0] + hx2[1].flip(-1)
            hx3 = hx3[0] + hx3[1].flip(-1)
            hx4 = hx4[0] + hx4[1].flip(-1)
            hx5 = hx5[0] + hx5[1].flip(-1)
            hx6 = hx6[0] + hx6[1].flip(-1)

            hx1 = F.interpolate(hx1.unsqueeze(0), x_size, mode='bilinear', align_corners=False)[0]
            hx2 = F.interpolate(hx2.unsqueeze(0), x_size, mode='bilinear', align_corners=False)[0]
            hx3 = F.interpolate(hx3.unsqueeze(0), x_size, mode='bilinear', align_corners=False)[0]
            hx4 = F.interpolate(hx4.unsqueeze(0), x_size, mode='bilinear', align_corners=False)[0]
            hx5 = F.interpolate(hx5.unsqueeze(0), x_size, mode='bilinear', align_corners=False)[0]
            hx6 = F.interpolate(hx6.unsqueeze(0), x_size, mode='bilinear', align_corners=False)[0]

            hx0 = torch.sum(torch.stack([hx1, hx2, hx3, hx4, hx5, hx6]), 0)

            return hx0, hx1, hx2, hx3, hx4, hx5, hx6


if __name__ == '__main__':
    from torchsummary import summary

    model = CAM()

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    y0, y1, y2, y3, y4, y5, y6 = y

    assert y0.shape == (20, 320, 320)
    assert y1.shape == (20, 320, 320)
    assert y2.shape == (20, 320, 320)
    assert y3.shape == (20, 320, 320)
    assert y4.shape == (20, 320, 320)
    assert y5.shape == (20, 320, 320)
    assert y6.shape == (20, 320, 320)

    model = CAM(mid_ch=32)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    y0, y1, y2, y3, y4, y5, y6 = y

    assert y0.shape == (20, 320, 320)
    assert y1.shape == (20, 320, 320)
    assert y2.shape == (20, 320, 320)
    assert y3.shape == (20, 320, 320)
    assert y4.shape == (20, 320, 320)
    assert y5.shape == (20, 320, 320)
    assert y6.shape == (20, 320, 320)

    model = CAM(mid_ch=64)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    y0, y1, y2, y3, y4, y5, y6 = y

    assert y0.shape == (20, 320, 320)
    assert y1.shape == (20, 320, 320)
    assert y2.shape == (20, 320, 320)
    assert y3.shape == (20, 320, 320)
    assert y4.shape == (20, 320, 320)
    assert y5.shape == (20, 320, 320)
    assert y6.shape == (20, 320, 320)
