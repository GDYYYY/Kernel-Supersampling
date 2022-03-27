
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
from model import ImportanceMap
from model import KernelConstruction
from model import RepVGG
from model import Supersampling
import time
import torch
import torch.nn as nn
import math

def make_model(args, parent=False):
    return KernelSR(args)

class KernelSR(nn.Module):
    def __init__(self, args):
        super(KernelSR, self).__init__()
        self.feat_extraction = RepVGG.RepVGGFE()
        self.importance_map = ImportanceMap.ImportanceMap(map_layers = 18)
        self.kernel_construction = KernelConstruction.KernelConstruction()
        self.supersampling = Supersampling.Supersampling()

    def forward(self, x):
        feat = self.feat_extraction(x)
        # print(feat.shape)
        feat, immap = self.importance_map(feat)
        # print(immap.shape)
        kernels = self.kernel_construction(immap)
        # print(kernels.shape)
        out = self.supersampling(feat,kernels)
        # print(out.shape)
        return out
