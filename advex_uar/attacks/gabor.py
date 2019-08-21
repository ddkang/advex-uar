import argparse
import itertools
import math
import numbers
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True

def normalize(orig):
    batch_size = orig.size(0)
    omax = torch.max(orig.view(batch_size, -1), 1)[0].detach().view(batch_size,1,1,1)
    omin  = torch.min(orig.view(batch_size, -1), 1)[0].detach().view(batch_size,1,1,1)
    return (orig - omin) / (omax - omin)

def get_gabor(k_size, sigma, Lambda, theta):
    y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, k_size), torch.linspace(-0.5, 0.5, k_size)])
    rotx = x * torch.cos(theta) + y * torch.sin(theta)
    roty = -x * torch.sin(theta) + y * torch.cos(theta)
    g = torch.zeros(y.shape)
    g = torch.exp(-0.5 * (rotx ** 2 / (sigma + 1e-3) ** 2 + roty ** 2 / (sigma + 1e-3) ** 2))
    g = g * torch.cos(2 * np.pi * Lambda * rotx)
    return g

def get_gabor_with_sides(k_size, sigma, Lambda, theta, sides=3):
    g = get_gabor(k_size, sigma, Lambda, theta)
    for i in range(1, sides):
        g += get_gabor(k_size, sigma, Lambda, theta + np.pi * i / sides)
    return g

def normalize_var(orig):
    batch_size = orig.size(0)

    # Spectral variance
    mean = torch.mean(orig.view(batch_size, -1), 1).view(batch_size, 1, 1, 1)
    spec_var = torch.rfft(torch.pow(orig -  mean, 2), 2)

    # Normalization
    imC = torch.sqrt(torch.irfft(spec_var, 2, signal_sizes=orig.size()[2:]).abs())
    imC /= torch.max(imC.view(batch_size, -1), 1)[0].view(batch_size, 1, 1, 1)
    minC = 0.001
    imK =  (minC + 1) / (minC + imC)

    mean, imK = mean.detach(), imK.detach()
    img = mean + (orig -  mean) * imK
    return normalize(img)

def gabor_rand_distributed(sp_conv, gabor_kernel):
    # code converted from https://github.com/kenny-co/procedural-advml

    batch_size = sp_conv.size(0)
    # reshape batch dimension to channel dimension to use group convolution
    # so that data processes in parallel
    sp_conv = sp_conv.view(1, batch_size, sp_conv.size(-2), sp_conv.size(-1))
    sp_conv = F.conv2d(sp_conv, weight=gabor_kernel, stride=1, groups=batch_size, padding=11)
    sp_conv = sp_conv.view(batch_size, 1, sp_conv.size(-2), sp_conv.size(-1))

    return normalize_var(sp_conv)
