import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


class Baseline(nn.Module):
    """
    baseline model, two maxpool
    """
    def __init__(self, n_class):
        super(Baseline, self).__init__()

        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=3, out_channels= 8, kernel_size=3, stride=1, padding=1,bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(8))

        self.layers.append(nn.Conv2d(in_channels=8, out_channels= 8, kernel_size=3, stride=1, padding=1,bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(2,2))
        self.layers.append(nn.BatchNorm2d(8))

        self.layers.append(nn.Conv2d(in_channels=8, out_channels= 16, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(16))
        self.layers.append(nn.Conv2d(in_channels=16, out_channels= 16, kernel_size=3, stride=1, padding=1,bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(2,2))
        self.layers.append(nn.BatchNorm2d(16))

        self.layers2 = []
        self.layers2.append(nn.Conv2d(in_channels=16, out_channels= 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers2.append(nn.ReLU(inplace=True))
        self.layers2.append(nn.BatchNorm2d(32))
        self.layers2.append(nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers2.append(nn.ReLU(inplace=True))
        self.layers2.append(nn.BatchNorm2d(32))
        self.layers2.append(nn.Upsample(scale_factor=2))

        self.layers3 = []
        self.layers3.append(nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers3.append(nn.ReLU(inplace=True))
        self.layers3.append(nn.BatchNorm2d(16))
        self.layers3.append(nn.Conv2d(in_channels=16, out_channels= 16, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers3.append(nn.ReLU(inplace=True))
        self.layers3.append(nn.BatchNorm2d(16))
        self.layers3.append(nn.Upsample(scale_factor=2))

        self.layers4 = []
        self.layers4.append(nn.Conv2d(in_channels=16, out_channels= 8, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers4.append(nn.ReLU(inplace=True))
        self.layers4.append(nn.BatchNorm2d(8))
        self.layers4.append(nn.Conv2d(in_channels=8, out_channels= 8, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers4.append(nn.ReLU(inplace=True))
        self.layers4.append(nn.Dropout(0.2))
        self.layers4.append(nn.Conv2d(in_channels=8, out_channels= n_class, kernel_size=1, stride=1, bias=False))                       
        self.layers4.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.layers)
        self.net2 = nn.Sequential(*self.layers2)
        self.net3 = nn.Sequential(*self.layers3)
        self.net4 = nn.Sequential(*self.layers4)

    def forward(self, x, evalMode=False):
        output = x # (B, 3, W, H)
        output = self.net(output)
        output = self.net2(output)
        output = self.net3(output)
        output = self.net4(output)
        return output