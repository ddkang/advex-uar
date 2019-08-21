import random

import numpy as np
import torch
import torch.nn as nn

from advex_uar.attacks.attacks import AttackWrapper
from advex_uar.attacks.fog import fog_creator

class FogAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False,
                 wibble_decay=2.0):
        """
        Parameters:
            nb_its (int):          Number of GD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            wibble_decay (float):  Fog-specific parameter
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.wibble_decay = wibble_decay

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its    

    def _init(self, batch_size, map_size):
        fog_vars = []
        for i in range(int(np.log2(map_size))):
            for j in range(3):
                var = torch.rand((batch_size, 2**i, 2**i), device="cuda")
                var.requires_grad_()
                fog_vars.append(var)
        return fog_vars
        
    def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
        pixel_inp = pixel_img.detach()
        batch_size = pixel_img.size(0)
        x_max, _ = torch.max(pixel_img.view(pixel_img.size(0), 3, -1), -1)
        x_max = x_max.view(-1, 3, 1, 1)
        map_size = 2 ** (int(np.log2(self.resol)) + 1)
        
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
        
        fog_vars = self._init(batch_size, map_size)
        fog = fog_creator(fog_vars, batch_size, mapsize=map_size,
                          wibbledecay=self.wibble_decay)[:,:,16:-16,16:-16]
        s = pixel_model(torch.clamp((pixel_inp + base_eps[:, None, None, None] * fog) /
                                    (x_max + base_eps[:, None, None, None]) * 255., 0., 255.))
        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                # to avoid the target, we increase the loss
                grads = [f.grad.data for f in fog_vars]
            else:
                # to hit the target, we reduce the loss
                grads = [-f.grad.data for f in fog_vars]

            grad_signs = [grad.sign() for grad in grads]
            for f, g in zip(fog_vars, grad_signs):
                f.data = f.data + step_size[:, None, None] * g
                f.detach()
                f.data = f.data.clamp(0, 1)
                f.requires_grad_()

            if it != self.nb_its - 1:
                fog = fog_creator(fog_vars, batch_size, mapsize=map_size,
                                  wibbledecay=self.wibble_decay)[:,:,16:-16,16:-16]
                s = pixel_model(torch.clamp((pixel_inp + base_eps[:, None, None, None] * fog) /
                                            (x_max + base_eps[:, None, None, None]) * 255., 0., 255.))
                for f in fog_vars:
                    f.grad.data.zero_()
        fog = fog_creator(fog_vars, batch_size, mapsize=map_size,
                          wibbledecay=self.wibble_decay)[:,:,16:-16,16:-16]
        pixel_result = torch.clamp((pixel_inp + base_eps[:, None, None, None] * fog) /
                                   (x_max + base_eps[:, None, None, None]) * 255., 0., 255.)
        return pixel_result
