from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


class UNetClassifier(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, threshold=0.15):
        super(UNetClassifier, self).__init__()

        features = init_features
        self.threshold = threshold
        self.out_channels = out_channels
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.classifier = nn.Conv2d(features * 16, out_channels, 1, bias=False)

    def forward(self, img):
        enc1 = self.encoder1(img)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        cls_label_pred = gap2d(bottleneck, keepdims=True)
        cls_label_pred = self.classifier(cls_label_pred)
        cls_label_pred = cls_label_pred.view(-1, self.out_channels)

        return cls_label_pred

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, threshold=0.15):
        super(UNet, self).__init__()

        features = init_features
        self.threshold = threshold
        self.out_channels = out_channels
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.classifier = nn.Conv2d(features * 16, 20, 1, bias=False)

    def forward_cls(self, img):
        enc1 = self.encoder1(img)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        cls_label_pred = gap2d(bottleneck, keepdims=True)
        cls_label_pred = self.classifier(cls_label_pred)
        cls_label_pred = cls_label_pred.view(-1, 20)

        return cls_label_pred

    def create_seg_label_from_cls_labels(self, bottleneck, y_true, seg_label):
        with torch.set_grad_enabled(False):
            x = F.conv2d(bottleneck.unsqueeze(0), self.classifier.weight)
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
                        seg_label_pseudo[seg_label == j] = i - 1

        return seg_label_pseudo.unsqueeze(0)

    def forward(self, img):
        enc1 = self.encoder1(img)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        cls_label_pred = gap2d(bottleneck, keepdims=True)
        cls_label_pred = self.classifier(cls_label_pred)
        cls_label_pred = cls_label_pred.view(-1, 20)

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
        seg_label_pred = torch.sigmoid(self.conv(dec1))
        return seg_label_pred, cls_label_pred, bottleneck

    def forward_bottleneck(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        return self.bottleneck(self.pool4(enc4))

    def forward_cam(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        x = F.conv2d(bottleneck, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == '__main__':
    model = UNet(out_channels=1)
    from torchsummary import summary

    summary(model, input_size=(3, 512, 512))
    x = torch.rand([2, 3, 512, 512])
    y, cls, bottleneck = model(x)
    cam = model.forward_cam(x)
    print(y.shape, bottleneck.shape, cls.shape, cam.shape)
    assert bottleneck.shape == (2, 512, 32, 32)
    assert y.shape == (2, 1, 512, 512)
    assert cls.shape == (2, 20)
    assert cam.shape == (20, 32, 32)
