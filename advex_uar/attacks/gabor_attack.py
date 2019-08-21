import random

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sparse

from advex_uar.attacks.attacks import AttackWrapper
from advex_uar.attacks.gabor import get_gabor_with_sides, valid_position, gabor_rand_distributed

class GaborAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False):
        """
        Parameters:
            nb_its (int):          Number of GD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its

    def _init(self, batch_size, num_kern):
        grid = 14

        if self.rand_init:
            sparse_matrices = []
            sp_conv_numpy = sparse.random(self.resol*batch_size, self.resol,
                            density= 1. / grid, format='csr')
            sp_conv_numpy.data = sp_conv_numpy.data * 2 - 1
            sp_conv = torch.FloatTensor(sp_conv_numpy.todense()).view(
                        batch_size, self.resol, self.resol)

            mask = (sp_conv == 0).cuda().float().view(-1, 1, self.resol, self.resol)
            gabor_vars = sp_conv.clone().cuda().view(-1, 1, self.resol, self.resol)
            gabor_vars.requires_grad_(True)
        return gabor_vars, mask

    def _get_gabor_kernel(self, batch_size):
        # make gabor filters to convolve with variables
        k_size = 23
        kernels = []
        for b in range(batch_size):
            sides = np.random.randint(10) + 1
            sigma = 0.3 * torch.rand(1) +  0.1
            Lambda = (k_size / 4. - 3) * torch.rand(1) + 3
            theta = np.pi * torch.rand(1)

            kernels.append(get_gabor_with_sides(k_size, sigma, Lambda, theta, sides).cuda())
        gabor_kernel = torch.cat(kernels, 0).view(-1, 1, k_size, k_size)
        return gabor_kernel

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

        gabor_kernel = self._get_gabor_kernel(batch_size)
        num_kern = np.random.randint(50) + 1
        gabor_vars, mask = self._init(batch_size, num_kern)
        gabor_noise = gabor_rand_distributed(gabor_vars, gabor_kernel)
        gabor_noise = gabor_noise.expand(-1, 3, -1, -1)
        s = pixel_model(torch.clamp(pixel_inp + base_eps[:, None, None, None] * gabor_noise, 0., 255.))
        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                # to avoid the target, we increase the loss
                grad = gabor_vars.grad.data
            else:
                # to hit the target, we reduce the loss
                grad = -gabor_vars.grad.data

            grad_sign = grad.sign()
            gabor_vars.data = gabor_vars.data + step_size[:, None, None, None] * grad_sign
            gabor_vars.data = torch.clamp(gabor_vars.data, -1, 1) * mask

            if it != self.nb_its - 1:
                gabor_noise = gabor_rand_distributed(gabor_vars, gabor_kernel).expand(-1, 3, -1, -1)
                s = pixel_model(torch.clamp(pixel_inp + base_eps[:, None, None, None] * gabor_noise, 0., 255.))
                gabor_vars.grad.data.zero_()
        pixel_result = torch.clamp(pixel_inp + base_eps[:, None, None, None] * gabor_rand_distributed(gabor_vars, gabor_kernel), 0., 255.)
        return pixel_result
