import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.u2net_layers import RSU7, RSU4, RSU5, RSU6, RSU4F, _upsample_like
from models.unet import gap2d


class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, threshold=0.15):
        super(U2NETP, self).__init__()
        self.threshold = threshold
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

        self.classifier = nn.Conv2d(64, 20, 1, bias=False)

    def _create_seg_label_from_cls_labels(self, bottleneck, y_true, seg_label):
        with torch.set_grad_enabled(False):
            x = F.conv2d(bottleneck, self.classifier.weight)
            x = F.relu(x).squeeze(0)
            valid_labels = torch.nonzero(y_true)[0, :]
            cams = F.interpolate(x.unsqueeze(0), seg_label.shape, mode='bilinear', align_corners=False).squeeze(0)
            cams = cams[valid_labels]

            cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5
            cams = cams.detach().cpu().numpy()

            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=self.threshold)

            keys = np.pad(valid_labels.cpu() + 1, (1, 0), mode='constant')

            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]

            seg_label_pseudo = torch.zeros(seg_label.shape, dtype=torch.long)
            for i in (valid_labels + 1).cpu().tolist():
                masks = seg_label[torch.eq(torch.from_numpy(cls_labels), i).to(seg_label.device)]
                segment_list = masks.unique()
                segment_list = segment_list[segment_list != 0]
                for j in segment_list:
                    active_area = (masks == j).sum()
                    segment_area = (seg_label == j).sum()
                    if segment_area < 1024:
                        continue
                    ratio = active_area.item() / segment_area.item()
                    if ratio > 0.04:
                        seg_label_pseudo[seg_label == j] = i

        return seg_label_pseudo.unsqueeze(0)

    def create_seg_label_from_cls_labels(self, bottleneck1, bottleneck2, bottleneck3, bottleneck4, bottleneck5,
                                         bottleneck6, cls_label, seg_label):
        with torch.set_grad_enabled(False):
            bottleneck1 = bottleneck1.unsqueeze(0)
            bottleneck2 = bottleneck2.unsqueeze(0)
            bottleneck3 = bottleneck3.unsqueeze(0)
            bottleneck4 = bottleneck4.unsqueeze(0)
            bottleneck5 = bottleneck5.unsqueeze(0)
            bottleneck6 = bottleneck6.unsqueeze(0)
            bottleneck2 = _upsample_like(bottleneck2, bottleneck1)
            bottleneck3 = _upsample_like(bottleneck3, bottleneck1)
            bottleneck4 = _upsample_like(bottleneck4, bottleneck1)
            bottleneck5 = _upsample_like(bottleneck5, bottleneck1)
            bottleneck6 = _upsample_like(bottleneck6, bottleneck1)
            bottleneck = bottleneck1 + bottleneck2 + bottleneck3 + bottleneck4 + bottleneck5 + bottleneck6
        return self._create_seg_label_from_cls_labels(bottleneck, cls_label, seg_label)

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
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        hx1_gap = gap2d(hx1, keepdims=True)
        hx2_gap = gap2d(hx2, keepdims=True)
        hx3_gap = gap2d(hx3, keepdims=True)
        hx4_gap = gap2d(hx4, keepdims=True)
        hx5_gap = gap2d(hx5, keepdims=True)
        hx6_gap = gap2d(hx6, keepdims=True)

        cls_lab1 = self.classifier(hx1_gap).view(-1, 20)
        cls_lab2 = self.classifier(hx2_gap).view(-1, 20)
        cls_lab3 = self.classifier(hx3_gap).view(-1, 20)
        cls_lab4 = self.classifier(hx4_gap).view(-1, 20)
        cls_lab5 = self.classifier(hx5_gap).view(-1, 20)
        cls_lab6 = self.classifier(hx6_gap).view(-1, 20)
        cls_lab0 = cls_lab1 + cls_lab2 + cls_lab3 + cls_lab4 + cls_lab5 + cls_lab6

        return (F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(
            d6)), (cls_lab0, cls_lab1, cls_lab2, cls_lab3, cls_lab4, cls_lab5, cls_lab6), (hx1, hx2, hx3, hx4, hx5, hx6)


class U2NETPClassifier(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, threshold=0.15, mid_ch=16):
        super(U2NETPClassifier, self).__init__()
        self.threshold = threshold
        self.stage1 = RSU7(in_ch, mid_ch, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, mid_ch, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, mid_ch, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, mid_ch, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, mid_ch, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, mid_ch, 64)

        # decoder
        self.stage5d = RSU4F(128, mid_ch, 64)
        self.stage4d = RSU4(128, mid_ch, 64)
        self.stage3d = RSU5(128, mid_ch, 64)
        self.stage2d = RSU6(128, mid_ch, 64)
        self.stage1d = RSU7(128, mid_ch, 64)

        self.classifier1 = nn.Conv2d(64, 20, 1, bias=False)
        self.classifier2 = nn.Conv2d(64, 20, 1, bias=False)
        self.classifier3 = nn.Conv2d(64, 20, 1, bias=False)
        self.classifier4 = nn.Conv2d(64, 20, 1, bias=False)
        self.classifier5 = nn.Conv2d(64, 20, 1, bias=False)
        self.classifier6 = nn.Conv2d(64, 20, 1, bias=False)

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

        cls_lab1 = self.classifier1(hx1_gap).view(-1, 20)
        cls_lab2 = self.classifier2(hx2_gap).view(-1, 20)
        cls_lab3 = self.classifier3(hx3_gap).view(-1, 20)
        cls_lab4 = self.classifier4(hx4_gap).view(-1, 20)
        cls_lab5 = self.classifier5(hx5_gap).view(-1, 20)
        cls_lab6 = self.classifier6(hx6_gap).view(-1, 20)
        cls_lab0 = cls_lab1 + cls_lab2 + cls_lab3 + cls_lab4 + cls_lab5 + cls_lab6

        return cls_lab0, cls_lab1, cls_lab2, cls_lab3, cls_lab4, cls_lab5, cls_lab6


if __name__ == '__main__':
    model = U2NETP()
    from torchsummary import summary

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y, cls, bottleneck = model(x)
    y0, y1, y2, y3, y4, y5, y6 = y
    cls0, cls1, cls2, cls3, cls4, cls5, cls6 = cls
    bottleneck1, bottleneck2, bottleneck3, bottleneck4, bottleneck5, bottleneck6 = bottleneck

    assert y0.shape == (2, 1, 320, 320)
    assert y1.shape == (2, 1, 320, 320)
    assert y2.shape == (2, 1, 320, 320)
    assert y3.shape == (2, 1, 320, 320)
    assert y4.shape == (2, 1, 320, 320)
    assert y5.shape == (2, 1, 320, 320)
    assert y6.shape == (2, 1, 320, 320)
    assert cls0.shape == (2, 20)
    assert cls1.shape == (2, 20)
    assert cls2.shape == (2, 20)
    assert cls3.shape == (2, 20)
    assert cls4.shape == (2, 20)
    assert cls5.shape == (2, 20)
    assert cls6.shape == (2, 20)

    assert bottleneck1.shape == (2, 64, 320, 320)
    assert bottleneck2.shape == (2, 64, 160, 160)
    assert bottleneck3.shape == (2, 64, 80, 80)
    assert bottleneck4.shape == (2, 64, 40, 40)
    assert bottleneck5.shape == (2, 64, 20, 20)
    assert bottleneck6.shape == (2, 64, 10, 10)

    model = U2NETPClassifier()
    from torchsummary import summary

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    y0, y1, y2, y3, y4, y5, y6 = y
    print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape, y5.shape, y6.shape)
