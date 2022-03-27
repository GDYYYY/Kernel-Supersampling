import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import math

class ImportanceMap(nn.Module): 

    def __init__(self, map_layers = 1):
        super(ImportanceMap, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.genmap = nn.Conv2d(in_channels=6, out_channels=map_layers, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        feat = self.upscale(x)
        out = self.genmap(feat)
        return feat, out

if __name__ == "__main__":
    input_feature = 12
    N = 8
    H = 224
    W = 224

    model = ImportanceMap(18)
    data = torch.ones((N,input_feature,H,W))

    model.cuda()
    data = data.cuda()
    print("data", data.shape)
    feat,out = model(data)
    print("feat", feat.shape)
    print("out", out.shape)