import torch
import torch.nn.functional as F

from models.unet.net import Net


class CAM(Net):

    def __init__(self, *args, **kwargs):
        super(CAM, self).__init__(*args, **kwargs)

    def forward(self, x):
        return self.forward_cam(x)

    def forward_cam(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        x = F.relu(F.conv2d(bottleneck, self.classifier.weight))
        x = x[0] + x[1].flip(-1)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = CAM()

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)
    print(y.shape)
    assert y.shape == (20, 20, 20)

    model = CAM(init_features=32)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)

    assert y.shape == (20, 20, 20)

    model = CAM(mid_ch=64)

    summary(model, input_size=(3, 320, 320))
    x = torch.rand([2, 3, 320, 320])
    y = model(x)

    assert y.shape == (20, 20, 20)
