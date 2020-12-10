import torch
import torch.nn as nn

class UNet_ConvTranspose2d(nn.Module):
    def __init__(self, n_class):
        super(UNet_ConvTranspose2d, self).__init__()

        # padding for same
        self.layer1 = nn.Sequential(
                            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(),
                            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU())
        self.layer2 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(
                            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU())
        self.layer4 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer5 = nn.Sequential(
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.layer6 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer7 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.layer8 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer9 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU())
        
        self.layer10 = nn.Sequential(
                            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
                            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.layer12 = nn.Sequential(
                            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2))
        self.layer13 = nn.Sequential(
                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.layer14 = nn.Sequential(
                            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2))
        self.layer15 = nn.Sequential(
                            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU())
        self.layer16 = nn.Sequential(
                            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2))
        self.layer17 = nn.Sequential(
                            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(),
                            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU())
        self.layer18 = nn.Sequential(
                            nn.Conv2d(8, n_class, kernel_size=1, stride=1))

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
                       self.layer6, self.layer7, self.layer8, self.layer9, self.layer10,
                       self.layer11, self.layer12, self.layer13, self.layer14, self.layer15,
                       self.layer16, self.layer17, self.layer18]

    def forward(self, x, evalMode=False):
        output = x
        out_name = ['c1', 'p1', 'c2', 'p2', 'c3', 'p3', 'c4', 'p4', 'c5', 
                    'u6', 'c6', 'u7', 'c7', 'u8', 'c8', 'u9', 'c9', 'd']
        out = dict()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            name = out_name[i]

            output = layer(output)
            if name == 'u6':
                output = torch.cat((output, out['c4']), dim=1)
            elif name == 'u7':
                output = torch.cat((output, out['c3']), dim=1)
            elif name == 'u8':
                output = torch.cat((output, out['c2']), dim=1)
            elif name == 'u9':
                output = torch.cat((output, out['c1']), dim=1)

            out[name] = output
        return output


class UNet_Upsample(nn.Module):
    def __init__(self, n_class):
        super(UNet_Upsample, self).__init__()

        # padding for same
        self.layer1 = nn.Sequential(
                            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(),
                            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU())
        self.layer2 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(
                            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU())
        self.layer4 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer5 = nn.Sequential(
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.layer6 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer7 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.layer8 = nn.Sequential(
                            nn.MaxPool2d(2,2))
        self.layer9 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(),
                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU())
        
        self.layer10 = nn.Sequential(
                            nn.Upsample(scale_factor=2))
        self.layer11 = nn.Sequential(
                            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
        self.layer12 = nn.Sequential(
                            nn.Upsample(scale_factor=2))
        self.layer13 = nn.Sequential(
                            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU())
        self.layer14 = nn.Sequential(
                            nn.Upsample(scale_factor=2))
        self.layer15 = nn.Sequential(
                            nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU())
        self.layer16 = nn.Sequential(
                            nn.Upsample(scale_factor=2))
        self.layer17 = nn.Sequential(
                            nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(),
                            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU())
        self.layer18 = nn.Sequential(
                            nn.Conv2d(8, n_class, kernel_size=1, stride=1))

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
                       self.layer6, self.layer7, self.layer8, self.layer9, self.layer10,
                       self.layer11, self.layer12, self.layer13, self.layer14, self.layer15,
                       self.layer16, self.layer17, self.layer18]

    def forward(self, x, evalMode=False):
        output = x
        out_name = ['c1', 'p1', 'c2', 'p2', 'c3', 'p3', 'c4', 'p4', 'c5', 
                    'u6', 'c6', 'u7', 'c7', 'u8', 'c8', 'u9', 'c9', 'd']
        out = dict()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            name = out_name[i]

            output = layer(output)
            if name == 'u6':
                output = torch.cat((output, out['c4']), dim=1)
            elif name == 'u7':
                output = torch.cat((output, out['c3']), dim=1)
            elif name == 'u8':
                output = torch.cat((output, out['c2']), dim=1)
            elif name == 'u9':
                output = torch.cat((output, out['c1']), dim=1)

            out[name] = output
        return output