import random

import torch
import torch.nn as nn

from advex_uar.attacks.attacks import AttackWrapper
from advex_uar.attacks.jpeg import JPEG

class JPEGAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol,
                 rand_init=True, opt='linf', scale_each=False, l1_max=2.):
        '''
        Arguments:
            nb_its (int):          Number of iterations
            eps_max (float):       Maximum flow, in L_inf norm, in pixels
            step_size (float):     Maximum step size in L_inf norm, in pixels
            resol (int):           Side length of images, in pixels
            rand_init (bool):      Whether to do a random init
            opt (string):          Which optimization algorithm to use, either 'linf', 'l1', or 'l2'
            scale_each (bool):     Whether to scale eps for each image in a batch separately
        '''
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.rand_init = rand_init
        self.opt = opt
        if opt not in ['linf', 'l1', 'l2']:
            raise NotImplementedError
        self.scale_each = scale_each
        self.l1_max = l1_max
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = nb_its
        self.jpeg = JPEG().cuda()

    def _convert_cat_var(self, cat_var, batch_size, height, width):
        y_var = cat_var[:, :height//8 * width//8 * 8 * 8].view((batch_size, height//8 * width//8, 8, 8))
        cb_var = cat_var[:, height//8 * width//8 * 8 * 8: height//8 * width//8 * 8 * 8 + height//16 * width//16 * 8 * 8].view((batch_size, height//16 * width//16, 8, 8))
        cr_var = cat_var[:, height//8 * width//8 * 8 * 8 + height//16 * width//16 * 8 * 8: height//8 * width//8 * 8 * 8 + 2 * height//16 * width//16 * 8 * 8].view((batch_size, height//16 * width//16, 8, 8))
        return y_var, cb_var, cr_var
    
    def _jpeg_cat(self, pixel_inp, cat_var, base_eps, batch_size, height, width):
        y_var, cb_var, cr_var = self._convert_cat_var(cat_var, batch_size, height, width)
        return self.jpeg(pixel_inp, [y_var, cb_var, cr_var], base_eps)
        
    def _run_one_pgd(self, pixel_model, pixel_inp, cat_var, target, base_eps, step_size, avoid_target=True):
        batch_size, channels, height, width = pixel_inp.size()
        pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var, base_eps, batch_size, height, width)
        s = pixel_model(pixel_inp_jpeg)

        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()

            if avoid_target:
                grad = cat_var.grad.data
            else:
                grad = -cat_var.grad.data

            if self.opt == 'linf':
                grad_sign = grad.sign()
                cat_var.data = cat_var.data + step_size[:, None] * grad_sign
                cat_var.data = torch.max(torch.min(cat_var.data, base_eps[:, None]),
                                         -base_eps[:, None]) # cannot use torch.clamp for tensors
            elif self.opt == 'l2':
                batch_size = pixel_inp.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None]                
                cat_var.data = cat_var.data + step_size[:, None] * normalized_grad
                l2_delta = torch.norm(cat_var.data.view(batch_size, -1), 2.0, dim=1)
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), base_eps / l2_delta)
                cat_var.data *= proj_scale[:, None]
                cat_var.data = torch.clamp(cat_var.data, -self.l1_max, self.l1_max)
                    
            if it != self.nb_its - 1:
                # self.jpeg scales rounding_vars by base_eps, so we divide to rescale
                # its coordinates to [-1, 1]
                cat_var_temp = cat_var / base_eps[:, None]
                pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var_temp, base_eps, batch_size, height, width)
                s = pixel_model(pixel_inp_jpeg)
            cat_var.grad.data.zero_()
        return cat_var
        
    def _run_one_fw(self, pixel_model, pixel_inp, cat_var, target, base_eps, avoid_target=True):
        batch_size, channels, height, width = pixel_inp.size()
        pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var, base_eps, batch_size, height, width)
        s = pixel_model(pixel_inp_jpeg)

        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()

            if avoid_target:
                grad = cat_var.grad.data
            else:
                grad = -cat_var.grad.data

            def where_float(cond, if_true, if_false):
                return cond.float() * if_true + (1-cond.float()) * if_false

            def where_long(cond, if_true, if_false):
                return cond.long() * if_true + (1-cond.long()) * if_false

            abs_grad = torch.abs(grad).view(batch_size, -1)
            num_pixels = abs_grad.size()[1]
            sign_grad = torch.sign(grad)

            bound = where_float(sign_grad > 0, self.l1_max - cat_var, cat_var + self.l1_max).view(batch_size, -1)
                
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
                
            delta_it = sign_grad * magnitudes.view(cat_var.size())
            # These should always be exactly epsilon
            # l1_check = torch.norm(delta_it.view(batch_size, -1), 1.0, dim=1) / num_pixels
            # print('l1_check: %s' % l1_check)
            cat_var.data = cat_var.data + (delta_it - cat_var.data) / (it + 1.0)

            if it != self.nb_its - 1:
                # self.jpeg scales rounding_vars by base_eps, so we divide to rescale
                # its coordinates to [-1, 1]
                cat_var_temp = cat_var / base_eps[:, None]
                pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var_temp, base_eps, batch_size, height, width)
                s = pixel_model(pixel_inp_jpeg)
            cat_var.grad.data.zero_()
        return cat_var

    def _init_empty(self, batch_size, height, width):
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        return torch.zeros(shape, device='cuda')
    
    def _init_linf(self, batch_size, height, width):
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        return torch.rand(shape, device='cuda') * 2 - 1
        
    def _init_l1(self, batch_size, height, width):
        # returns random initialization with L_1 norm of 1
        # algorithm from https://arxiv.org/abs/math/0503650, Theorem 1
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        exp = torch.empty(shape, dtype=torch.float32, device='cuda')
        exp.exponential_()
        signs = torch.sign(torch.randn(shape, dtype=torch.float32, device='cuda'))
        exp = exp * signs
        exp_y = torch.empty(shape[0], dtype=torch.float32, device='cuda')
        exp_y.exponential_()
        norm = exp_y + torch.norm(exp.view(shape[0], -1), 1.0, dim=1)
        init = exp / norm[:, None]
        return init

    def _init_l2(self, batch_size, height, width):
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        init = torch.randn(shape, dtype=torch.float32, device='cuda')
        init_norm = torch.norm(init.view(batch_size, -1), 2.0, dim=1)
        normalized_init = init / init_norm[:, None]
        rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/shape[1])
        init = normalized_init * rand_norms[:, None]
        return init
        
    def _init(self, batch_size, height, width, eps):
        if self.rand_init:
            if self.opt == 'linf':
                cat_var = self._init_linf(batch_size, height, width)
            elif self.opt == 'l1':
                cat_var = self._init_l1(batch_size, height, width)
            elif self.opt == 'l2':
                cat_var = self._init_l2(batch_size, height, width)
            else:
                raise NotImplementedError
        else:
            cat_var = self._init_empty(batch_size, height, width)
        cat_var.mul_(eps[:, None])
        cat_var.requires_grad_()
        return cat_var

    def _forward(self, pixel_model, pixel_img, target, scale_eps=False, avoid_target=False):
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
        batch_size, channels, height, width = pixel_img.size()
        if height % 16 != 0 or width % 16 != 0:
            raise Exception
        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True
        cat_var = self._init(batch_size, height, width, base_eps)
        if self.nb_its > 0:
            if self.opt in ['linf', 'l2']:
                cat_var = self._run_one_pgd(pixel_model, pixel_inp, cat_var, target,
                                            base_eps, step_size, avoid_target=avoid_target)
            elif self.opt == 'l1':
                cat_var = self._run_one_fw(pixel_model, pixel_inp, cat_var, target,
                                           base_eps, avoid_target=avoid_target)
            else:
                raise NotImplementedError
        # self.jpeg scales rounding_vars by base_eps, so we divide to rescale
        # its coordinates to [-1, 1]
        cat_var_temp = cat_var / base_eps[:, None]
        pixel_result = self._jpeg_cat(pixel_inp, cat_var_temp, base_eps, batch_size, height, width)
        return pixel_result

