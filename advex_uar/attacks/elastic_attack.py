import random

import torch
import torch.nn as nn

from advex_uar.attacks.attacks import AttackWrapper
from advex_uar.attacks.elastic import ElasticDeformation

class ElasticAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol,
                 rand_init=True, scale_each=False,
                 kernel_size=25, kernel_std=3):
        '''
        Arguments:
            nb_its (int):          Number of iterations
            eps_max (float):       Maximum flow, in L_inf norm, in pixels
            step_size (float):     Maximum step size in L_inf norm, in pixels
            resol (int):           Side length of images, in pixels
            rand_init (bool):      Whether to do a random init
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            kernel_size (int):     Size, in pixels of gaussian kernel
            kernel_std (int):      Standard deviation of kernel
        '''
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.deformer = ElasticDeformation(resol, kernel_size, kernel_std)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its

    def _init(self, batch_size, eps):
        if self.rand_init:
            # initialized randomly in [-1, 1], then scaled to [-base_eps, base_eps]
            flow = torch.rand((batch_size, 2, self.resol, self.resol),
                              dtype=torch.float32, device='cuda') * 2 - 1
            flow = eps[:, None, None, None] * flow
        else:
            flow = torch.zeros((batch_size, 2, self.resol, self.resol),
                               dtype=torch.float32, device='cuda')
        flow.requires_grad_()
        return flow
        
    def _forward(self, pixel_model, pixel_img, target, scale_eps=False, avoid_target=True):
        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True

        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand * self.eps_max
            step_size = rand * self.step_size
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        # Our base_eps and step_size are in pixel scale, but flow is in [-1, 1] scale
        base_eps.mul_(2.0 / self.resol)
        step_size.mul_(2.0 / self.resol)

        flow = self._init(pixel_img.size()[0], base_eps)        
        pixel_inp_adv = self.deformer(pixel_inp, flow)

        if self.nb_its > 0:
            res = pixel_model(pixel_inp_adv)       
            for it in range(self.nb_its):
                loss = self.criterion(res, target)
                loss.backward()

                if avoid_target:
                    grad = flow.grad.data
                else:
                    grad = -flow.grad.data
                
                # step_size has already been converted to [-1, 1] scale
                flow.data = flow.data + step_size[:, None, None, None] * grad.sign()
                flow.data = torch.max(torch.min(flow.data, base_eps[:, None, None, None]), -base_eps[:, None, None, None])            
                pixel_inp_adv = self.deformer(pixel_inp, flow)
                if it != self.nb_its - 1:
                    res = pixel_model(pixel_inp_adv)
                    flow.grad.data.zero_()
        return pixel_inp_adv

