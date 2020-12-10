import torch
import torch.nn as nn

class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, predicted, target, smooth=1):
        y_pred = torch.flatten(predicted)
        y_true = torch.flatten(target)
        
        intersection = torch.sum(y_true * y_pred)
        
        return -2*(intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)