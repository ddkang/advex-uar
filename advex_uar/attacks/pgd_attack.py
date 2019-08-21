import random

import numpy as np
import torch
import torch.nn as nn

from advex_uar.attacks.attacks import AttackWrapper

class PGDAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol, norm='linf', rand_init=True, scale_each=False):
        """
        Parameters:
            nb_its (int):          Number of PGD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            norm (str):            Either 'linf' or 'l2'
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its
        
    def _run_one(self, pixel_model, pixel_inp, delta, target, eps, step_size, avoid_target=True):
        s = pixel_model(pixel_inp + delta)
        if self.norm == 'l2':
            l2_max = eps
        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                # to avoid the target, we increase the loss
                grad = delta.grad.data
            else:
                # to hit the target, we reduce the loss
                grad = -delta.grad.data

            if self.norm == 'linf':
                grad_sign = grad.sign()
                delta.data = delta.data + step_size[:, None, None, None] * grad_sign
                delta.data = torch.max(torch.min(delta.data, eps[:, None, None, None]), -eps[:, None, None, None])
                delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
            elif self.norm == 'l2':
                batch_size = delta.data.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None, None, None]                
                delta.data = delta.data + step_size[:, None, None, None] * normalized_grad
                l2_delta = torch.norm(delta.data.view(batch_size, -1), 2.0, dim=1)
                # Check for numerical instability
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), l2_max / l2_delta)
                delta.data *= proj_scale[:, None, None, None]
                delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
            else:
                raise NotImplementedError

            if it != self.nb_its - 1:
                s = pixel_model(pixel_inp + delta)
                delta.grad.data.zero_()
        return delta

    def _init(self, shape, eps):
        if self.rand_init:
            if self.norm == 'linf':
                init = torch.rand(shape, dtype=torch.float32, device='cuda') * 2 - 1
            elif self.norm == 'l2':
                init = torch.randn(shape, dtype=torch.float32, device='cuda')
                init_norm = torch.norm(init.view(init.size()[0], -1), 2.0, dim=1)
                normalized_init = init / init_norm[:, None, None, None]
                dim = init.size()[1] * init.size()[2] * init.size()[3]
                rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/dim)
                init = normalized_init * rand_norms[:, None, None, None]
            else:
                raise NotImplementedError
            init = eps[:, None, None, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')
    
    def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand.mul(self.eps_max)
            step_size = rand.mul(self.step_size)
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True
        delta = self._init(pixel_inp.size(), base_eps)
        if self.nb_its > 0:
            delta = self._run_one(pixel_model, pixel_inp, delta, target, base_eps,
                                  step_size, avoid_target=avoid_target)
        else:
            delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
        pixel_result = pixel_inp + delta
        return pixel_result

