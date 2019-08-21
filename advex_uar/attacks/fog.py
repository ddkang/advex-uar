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

def fog_creator(fog_vars, bsize=1, mapsize=256, wibbledecay=1.75):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = torch.from_numpy(np.empty((bsize, mapsize, mapsize), dtype=np.float32)).cuda()
    maparray[:, 0, 0] = 0
    stepsize = mapsize
    wibble = 100
    
    var_num = 0
    
    def wibbledmean(array, var_num):
        result = array / 4. + fog_vars[var_num] * 2 * wibble - wibble
        return result
    
    def fillsquares(var_num):
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[:, 0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + torch.roll(cornerref, -1, 1)
        squareaccum = squareaccum + torch.roll(squareaccum, -1, 2)
        maparray[:, stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum, var_num)
        return var_num + 1

    def filldiamonds(var_num):
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.size(1)
        drgrid = maparray[:, stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[:, 0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + torch.roll(drgrid, 2, 1)
        lulsum = ulgrid + torch.roll(ulgrid, -1, 2)
        ltsum = ldrsum + lulsum
        maparray[:, 0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum, var_num)
        var_num += 1
        tdrsum = drgrid + torch.roll(drgrid, 2, 2)
        tulsum = ulgrid + torch.roll(ulgrid, -1, 1)
        ttsum = tdrsum + tulsum
        maparray[:, stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum, var_num)
        return var_num + 1

    while stepsize >= 2:
        var_num = fillsquares(var_num)
        var_num = filldiamonds(var_num)
        stepsize //= 2
        wibble /= wibbledecay

    maparray = maparray - maparray.min()
    return (maparray / maparray.max()).reshape(bsize, 1, mapsize, mapsize)
