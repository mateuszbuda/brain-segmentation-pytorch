import torch
import torch.nn.functional as F

from models.layers import gap2d
from models.unet.cam import CAM


class Segmentation(CAM):

    def __init__(self, *args, **kwargs):
        super(Segmentation, self).__init__(*args, **kwargs)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        cls_label_pred = gap2d(bottleneck, keepdims=True)
        cls_label_pred = self.classifier(cls_label_pred)
        cls_label_pred = cls_label_pred.view(-1, self.num_classes)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        seg_label_pred = self.conv(dec1)
        return cls_label_pred, seg_label_pred

    def generate_pseudo_label(self, imgs, cls_label_true, original_image_size, cam_order=0, *args, **kwargs):
        with torch.set_grad_enabled(False):
            selected_cam = [self.forward_cam(img[0]) for img in imgs]
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
    assert cls_pred.shape == (2, 20)
    assert seg_pred.shape == (2, 1, 320, 320)

    model = Segmentation(mid_ch=32)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    cls_pred, seg_pred = model(x)
    assert cls_pred.shape == (2, 20)
    assert seg_pred.shape == (2, 1, 320, 320)

    model = Segmentation(mid_ch=64)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    cls_pred, seg_pred = model(x)
    assert cls_pred.shape == (2, 20)
    assert seg_pred.shape == (2, 1, 320, 320)

    ## Test Generating PSEUDO Labels
    imgs = torch.rand([1, 1, 2, 3, 320, 320])
    cam = model.generate_pseudo_label(imgs, torch.Tensor([1, 0, 0, 1, 0, 0, 0]), (512, 512))
    assert cam.shape == (2, 512, 512)
