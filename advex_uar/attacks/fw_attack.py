import random

import torch
import torch.nn as nn

from advex_uar.attacks.attacks import AttackWrapper

class FrankWolfeAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, resol, rand_init=True, scale_each=False):
        """
        Parameters:
            nb_its (int):          Number of FW iterations.
            eps_max (float):       The max norm, in pixel space (out of 255). Total L1 norm is (resol * resol * 3 * eps_max)
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to use a random init in the l1 ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        num_pixels = resol * resol * 3
        self.l1_max = eps_max * num_pixels
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.nb_backward_steps = self.nb_its

    def _run_one(self, pixel_model, pixel_inp, delta, target, base_eps, avoid_target=True):
        s = pixel_model(pixel_inp + delta)
        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                grad = delta.grad.data
            else:
                grad = - delta.grad.data
            
            def where_float(cond, if_true, if_false):
                return cond.float() * if_true + (1-cond.float()) * if_false

            def where_long(cond, if_true, if_false):
                return cond.long() * if_true + (1-cond.long()) * if_false


            # shape of grad: (batch, c, x, y)
            batch_size = grad.size()[0]
            abs_grad = torch.abs(grad).view(batch_size, -1)
            num_pixels = abs_grad.size()[1]
            sign_grad = torch.sign(grad)

            bound = where_float(sign_grad > 0, 255 - pixel_inp, pixel_inp).view(batch_size, -1)
                
            k_min = torch.zeros((batch_size,1), dtype=torch.long, requires_grad=False, device='cuda')
            k_max = torch.ones((batch_size,1), dtype=torch.long, requires_grad=False, device='cuda') * num_pixels
                
            # cum_bnd[k] is meant to track the L1 norm we end up with if we take 
            # the k indices with the largest gradient magnitude and push them to their boundary values (0 or 255)
            values, indices = torch.sort(abs_grad, descending=True)
            bnd = torch.gather(bound, 1, indices)
            # subtract bnd because we don't want the cumsum to include the final element
            cum_bnd = torch.cumsum(bnd, 1) - bnd
                
            # this is hard-coded as floor(log_2(256 * 256 * 3))
            for _ in range(17):
                k_mid = (k_min + k_max) // 2
                l1norms = torch.gather(cum_bnd, 1, k_mid)
                k_min = where_long(l1norms > base_eps, k_min, k_mid)
                k_max = where_long(l1norms > base_eps, k_mid, k_max)
                
            # next want to set the gradient of indices[0:k_min] to their corresponding bound
            magnitudes = torch.zeros((batch_size, num_pixels), requires_grad=False, device='cuda')
            for bi in range(batch_size):
                magnitudes[bi, indices[bi, :k_min[bi,0]]] = bnd[bi, :k_min[bi,0]]
                magnitudes[bi, indices[bi, k_min[bi,0]]] = base_eps[bi] - cum_bnd[bi, k_min[bi,0]]
                
            delta_it = sign_grad * magnitudes.view(pixel_inp.size())
            # These should always be exactly epsilon
            # l1_check = torch.norm(delta_it.view(batch_size, -1), 1.0, dim=1) / num_pixels
            # print('l1_check: %s' % l1_check)
            delta.data = delta.data + (delta_it - delta.data) / (it + 1.0)
            # These will generally be a bit smaller than epsilon
            # l1_check2 = torch.norm(delta.data.view(batch_size, -1), 1.0, dim=1) / num_pixels
            # print('l1_check2: %s' % l1_check2)

            if it != self.nb_its - 1:
                s = pixel_model(pixel_inp + delta)
            delta.grad.data.zero_()
        return delta

    def _init(self, shape, eps):
        if self.rand_init:
            # algorithm from https://arxiv.org/abs/math/0503650, Theorem 1
            exp = torch.empty(shape, dtype=torch.float32, device='cuda')
            exp.exponential_()
            signs = torch.sign(torch.randn(shape, dtype=torch.float32, device='cuda'))
            exp = exp * signs
            exp_y = torch.empty(shape[0], dtype=torch.float32, device='cuda')
            exp_y.exponential_()
            norm = exp_y + torch.norm(exp.view(shape[0], -1), 1.0, dim=1)
            init = exp / norm[:, None, None, None]
            init = eps[:, None, None, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')
    
    def _forward(self, pixel_model, pixel_img, target, scale_eps=False, avoid_target=True):
        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand * self.l1_max
        else:
            base_eps = self.l1_max * torch.ones(pixel_img.size()[0], device='cuda')

        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True
        delta = self._init(pixel_inp.size(), base_eps)
        if self.nb_its > 0:
            delta = self._run_one(pixel_model, pixel_inp, delta, target,
                                  base_eps, avoid_target=avoid_target)
        else:
            delta.data = torch.clamp(pixel_inp.data + delta.data, 0., 255.) - pixel_inp.data
        pixel_result = pixel_inp + delta
        return pixel_result

