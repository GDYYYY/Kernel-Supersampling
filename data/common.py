import random

import numpy as np
import skimage.color as sc

import torch
from torchvision import transforms

def get_patch(*args, patch_size=96, scale=1, multi_scale=False):
    ih, iw = args[0].shape[:2]
    # print("ih:",ih)
    # print("iw:",iw)
    # print("patch_size:",patch_size)
    # print("scale:",scale)

    # multi_scale = True
    if multi_scale:
        tp = (int)(int(scale)* int(patch_size))
        ip = patch_size
    else:
        tp =  (int)(int(scale)* int(patch_size))
        ip = patch_size
    # print("tp:",tp)
    # print("ip:",ip)
    #ix = random.randrange(0, iw - ip + 1)
    #iy = random.randrange(0, ih - ip + 1)
    #ix = random.randrange(0, (iw-ip)//(scale*10))*scale*10
    #iy  = random.randrange(0, (ih-ip)//(scale*10))*scale*10
    if scale==int(scale):
        step = 1
    elif (scale*2)== int(scale*2):
        step = 2
    elif (scale*5) == int(scale*5):
        step = 5
    else:
        step = 10

    ix = random.randrange(0, (iw-ip)//step)*step
    iy = random.randrange(0, (ih-ip)//step) *step
    # print("ix:",ix)
    # print("iy:",iy)
    tx, ty = int(int(scale) * ix), int(int(scale)  * iy)
    # print("tx:",tx)
    # print("ty:",ty)
    # print("args[1:]:",args[1:])
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]

