from collections import OrderedDict

import torch
import torch.nn as nn

from models.layers import gap2d


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


class Net(nn.Module):

    def __init__(self, in_ch=3, mid_ch=32, out_ch=1, num_classes=20, *args, **kwargs):
        super(Net, self).__init__()

        self.out_channels = out_ch
        self.num_classes = num_classes
        self.encoder1 = _block(in_ch, mid_ch, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(mid_ch, mid_ch * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(mid_ch * 2, mid_ch * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(mid_ch * 4, mid_ch * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(mid_ch * 8, mid_ch * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            mid_ch * 16, mid_ch * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((mid_ch * 8) * 2, mid_ch * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            mid_ch * 8, mid_ch * 4, kernel_size=2, stride=2
        )
        self.decoder3 = _block((mid_ch * 4) * 2, mid_ch * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            mid_ch * 4, mid_ch * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((mid_ch * 2) * 2, mid_ch * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            mid_ch * 2, mid_ch, kernel_size=2, stride=2
        )
        self.decoder1 = _block(mid_ch * 2, mid_ch, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=mid_ch, out_channels=out_ch, kernel_size=1
        )

        self.classifier = nn.Conv2d(mid_ch * 16, num_classes, 1, bias=False)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        cls_label_pred = gap2d(bottleneck, keepdims=True)
        cls_label_pred = self.classifier(cls_label_pred)
        cls_label_pred = cls_label_pred.view(-1, self.num_classes)

        return cls_label_pred


if __name__ == '__main__':
    from torchsummary import summary

    model = Net()

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    print(y.shape)
    assert y.shape == (2, 20)

    model = Net(mid_ch=32)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)

    assert y.shape == (2, 20)

    model = Net(mid_ch=64)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)

    assert y.shape == (2, 20)
