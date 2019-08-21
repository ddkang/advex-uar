import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from advex_uar.attacks.attacks import AttackWrapper
from advex_uar.attacks.snow import snow_creator, make_kernels

def apply_snow(img, snow, scale, discolor=0.25):    
    out = (1 - discolor) * img +\
    discolor * torch.max(img, (0.2126 * img[:,0:1] + 0.7152 * img[:,1:2] + 0.0722 * img[:,2:3]) * 1.5 + 0.5)   
    return torch.clamp(out + scale[:, None, None, None] * snow, 0, 1)

class SnowAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False,
                 budget=0.2):
        """
        Parameters:
            nb_its (int):          Number of GD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            budget (float):        Controls rate parameter of snowflakes
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.budget = budget

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its    

    def _init(self, batch_size):
        # flake intensities follow an exponential distribution
        flake_intensities = torch.exp(-1./(self.budget)*torch.rand(batch_size,7,self.resol//4,self.resol//4)).cuda()
        flake_intensities.requires_grad_(True)
        
        return flake_intensities

    def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
        pixel_inp = pixel_img.detach()
        batch_size = pixel_img.size(0)
        
        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand.mul(self.eps_max)
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')
        
        flake_intensities = self._init(batch_size)
        kernels = make_kernels()
        snow = snow_creator(flake_intensities, kernels, self.resol)
        s = pixel_model(apply_snow(pixel_inp / 255., snow, base_eps) * 255)

        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''            
            if avoid_target:
                # to avoid the target, we increase the loss
                grad = flake_intensities.grad.data
            else:
                # to hit the target, we reduce the loss
                grad = -flake_intensities.grad.data

            grad_sign = grad.sign()
            flake_intensities.data = flake_intensities.data + step_size[:, None, None, None] * grad_sign

            if it != self.nb_its - 1:
                snow = snow_creator(flake_intensities, kernels, self.resol)
                s = pixel_model(apply_snow(pixel_inp / 255., snow, base_eps) * 255)
                flake_intensities.grad.data.zero_()

                # keep intensities in valid range
                flake_intensities.detach()
                flake_intensities.data = flake_intensities.data.clamp(1e-9, 1)

                # ensure intensities do not exceed budget
                block_size = 8
                blocks = flake_intensities.size(-1)//block_size

                budget_per_region = F.adaptive_avg_pool2d(flake_intensities, blocks)
                budget_per_region[budget_per_region < self.budget] = self.budget

                for i in range(blocks):
                    for j in range(blocks):
                        flake_intensities.data[
                            :,:,i*block_size:(i+1)*block_size,
                            j*block_size:(j+1)*block_size] *= self.budget/budget_per_region[:,:,i,j].view(-1,7,1,1)

                flake_intensities.requires_grad_()

        snow = snow_creator(flake_intensities, kernels, self.resol)
        pixel_result = apply_snow(pixel_inp / 255., snow, base_eps) * 255
        return pixel_result

