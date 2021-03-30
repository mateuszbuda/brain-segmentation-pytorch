import torch
import torch.nn.functional as F

from models.layers import gap2d
from models.u2net_layers import _upsample_like
from models.u2netp.cam import CAM


class Segmentation(CAM):

    def __init__(self, *args, **kwargs):
        super(Segmentation, self).__init__(*args, **kwargs)

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
        if self.share_classifier:
            cls_lab1 = self.classifier(hx1_gap).view(-1, self.num_classes)
            cls_lab2 = self.classifier(hx2_gap).view(-1, self.num_classes)
            cls_lab3 = self.classifier(hx3_gap).view(-1, self.num_classes)
            cls_lab4 = self.classifier(hx4_gap).view(-1, self.num_classes)
            cls_lab5 = self.classifier(hx5_gap).view(-1, self.num_classes)
            cls_lab6 = self.classifier(hx6_gap).view(-1, self.num_classes)
        else:
            cls_lab1 = self.classifier1(hx1_gap).view(-1, self.num_classes)
            cls_lab2 = self.classifier2(hx2_gap).view(-1, self.num_classes)
            cls_lab3 = self.classifier3(hx3_gap).view(-1, self.num_classes)
            cls_lab4 = self.classifier4(hx4_gap).view(-1, self.num_classes)
            cls_lab5 = self.classifier5(hx5_gap).view(-1, self.num_classes)
            cls_lab6 = self.classifier6(hx6_gap).view(-1, self.num_classes)
        cls_lab0 = cls_lab1 + cls_lab2 + cls_lab3 + cls_lab4 + cls_lab5 + cls_lab6
        # Returns class_predictions, segmentation_predictions
        return (cls_lab0, cls_lab1, cls_lab2, cls_lab3, cls_lab4, cls_lab5, cls_lab6), (d0, d1, d2, d3, d4, d5, d6)

    def generate_pseudo_label(self, imgs, cls_label_true, original_image_size, cam_order=0, *args, **kwargs):
        with torch.set_grad_enabled(False):
            cams = list(zip(*[self.forward_cam(img[0]) for img in imgs]))
            selected_cam = cams[cam_order]
            strided_cam = torch.sum(
                torch.stack(
                    [F.interpolate(o, original_image_size, mode='bilinear', align_corners=False)[0] for o in
                     selected_cam]),
                0)

            valid_cat = torch.nonzero(cls_label_true, as_tuple=False)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
            return strided_cam


if __name__ == '__main__':
    from torchsummary import summary

    model = Segmentation()

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    cls_pred, seg_pred = model(x)
    for i in cls_pred:
        assert i.shape == (2, 20)
    for i in seg_pred:
        assert i.shape == (2, 1, 320, 320)

    model = Segmentation(mid_ch=32)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    cls_pred, seg_pred = model(x)
    for i in cls_pred:
        assert i.shape == (2, 20)
    for i in seg_pred:
        assert i.shape == (2, 1, 320, 320)

    model = Segmentation(mid_ch=64)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    cls_pred, seg_pred = model(x)
    for i in cls_pred:
        assert i.shape == (2, 20)
    for i in seg_pred:
        assert i.shape == (2, 1, 320, 320)

    ## Test Generating PSEUDO Labels
    imgs = torch.rand([1, 1, 2, 3, 320, 320])
    cam = model.generate_pseudo_label(imgs, torch.Tensor([1, 0, 0, 1, 0, 0, 0]), (512, 512))
    assert cam.shape == (2, 512, 512)

    model = Segmentation(out_ch=21)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    cls_pred, seg_pred = model(x)
    for i in cls_pred:
        assert i.shape == (2, 20)
    for i in seg_pred:
        assert i.shape == (2, 21, 320, 320)
