import torch
import torch.nn as nn
import models
from models.Baseline import Baseline
from models.UNet import UNet_ConvTranspose2d, UNet_Upsample
from models.FCNs import FCN8s
from models.DAG import FCNs

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


def build_model(model_name, n_class=32):
    if model_name == 'Baseline':
        return Baseline(n_class)
    elif model_name == 'UNet_ConvTranspose2d':
        return UNet_ConvTranspose2d(n_class)
    elif model_name == 'UNet_Upsample':
        return UNet_Upsample(n_class)
    elif model_name == 'FCN8':
        encoder = models.FCNs.VGGNet(batch_norm=True)
        return FCN8s(encoder, n_class)
    elif model_name == 'DAG':
        vgg_model = models.DAG.VGGNet(pretrained=False, requires_grad=True)
        return FCNs(vgg_model, n_class)
