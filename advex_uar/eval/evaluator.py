import importlib
import os
import subprocess

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms, models

from advex_uar.common.loader import StridedImageFolder
from advex_uar.eval.cifar10c import CIFAR10C
from advex_uar.train.trainer import Metric, accuracy, correct

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def norm_to_pil_image(img):
    img_new = torch.Tensor(img)
    for t, m, s in zip(img_new, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    img_new.mul_(255)
    np_img = np.rollaxis(np.uint8(img_new.numpy()), 0, 3)
    return Image.fromarray(np_img, mode='RGB')

class Accumulator(object):
    def __init__(self, name):
        self.name = name
        self.vals = []

    def update(self, val):
        self.vals.append(val)

    @property
    def avg(self):
        total_sum = sum([torch.sum(v) for v in self.vals])
        total_size = sum([v.size()[0] for v in self.vals])
        return total_sum / total_size

class BaseEvaluator():
    def __init__(self, **kwargs):
        default_attr = dict(
            # eval options
            model=None, batch_size=32, stride=10,
            dataset_path=None, # val dir for imagenet, base dir for CIFAR-10-C
            nb_classes=None,
            # attack options
            attack=None,
            # Communication options
            fp16_allreduce=False,
            # Logging options
            logger=None)
        default_attr.update(kwargs)
        for k in default_attr:
            setattr(self, k, default_attr[k])
        if self.dataset not in ['imagenet', 'imagenet-c', 'cifar-10', 'cifar-10-c']:
            raise NotImplementedError
        self.cuda = True
        if self.cuda:
            self.model.cuda()
        self.attack = self.attack()
        self._init_loaders()

    def _init_loaders(self):
        raise NotImplementedError
        
    def evaluate(self):
        self.model.eval()

        std_loss = Accumulator('std_loss')
        adv_loss = Accumulator('adv_loss')
        std_corr = Accumulator('std_corr')
        adv_corr = Accumulator('adv_corr')
        std_logits = Accumulator('std_logits')
        adv_logits = Accumulator('adv_logits')

        seen_classes = []
        adv_images = Accumulator('adv_images')
        first_batch_images = Accumulator('first_batch_images')

        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(data)
                std_logits.update(output.cpu())
                loss = F.cross_entropy(output, target, reduction='none').cpu()
                std_loss.update(loss)
                corr = correct(output, target)
                corr = corr.view(corr.size()[0]).cpu()
                std_corr.update(corr)

            rand_target = torch.randint(
                0, self.nb_classes - 1, target.size(),
                dtype=target.dtype, device='cuda')
            rand_target = torch.remainder(target + rand_target + 1, self.nb_classes)
            data_adv = self.attack(self.model, data, rand_target,
                                   avoid_target=False, scale_eps=False)

            for idx in range(target.size()[0]):
                if target[idx].cpu() not in seen_classes:
                    seen_classes.append(target[idx].cpu())
                    orig_image = norm_to_pil_image(data[idx].detach().cpu())
                    adv_image = norm_to_pil_image(data_adv[idx].detach().cpu())
                    adv_images.update((orig_image, adv_image, target[idx].cpu()))

            if batch_idx == 0:
                for idx in range(target.size()[0]):
                    orig_image = norm_to_pil_image(data[idx].detach().cpu())
                    adv_image = norm_to_pil_image(data_adv[idx].detach().cpu())
                    first_batch_images.update((orig_image, adv_image))
                
            with torch.no_grad():
                output_adv = self.model(data_adv)
                adv_logits.update(output_adv.cpu())
                loss = F.cross_entropy(output_adv, target, reduction='none').cpu()
                adv_loss.update(loss)
                corr = correct(output_adv, target)
                corr = corr.view(corr.size()[0]).cpu()
                adv_corr.update(corr)

            run_output = {'std_loss':std_loss.avg,
                          'std_acc':std_corr.avg,
                          'adv_loss':adv_loss.avg,
                          'adv_acc':adv_corr.avg}
            print('Batch', batch_idx)
            print(run_output)
            if batch_idx % 20 == 0:
                self.logger.log(run_output, batch_idx)

        summary_dict = {'std_acc':std_corr.avg.item(),
                        'adv_acc':adv_corr.avg.item()}
        self.logger.log_summary(summary_dict)
        for orig_img, adv_img, target in adv_images.vals:
            self.logger.log_image(orig_img, 'orig_{}.png'.format(target))
            self.logger.log_image(adv_img, 'adv_{}.png'.format(target))
        for idx, imgs in enumerate(first_batch_images.vals):
            orig_img, adv_img = imgs
            self.logger.log_image(orig_img, 'init_orig_{}.png'.format(idx))
            self.logger.log_image(adv_img, 'init_adv_{}.png'.format(idx))

        self.logger.end()
        print(std_loss.avg, std_corr.avg, adv_loss.avg, adv_corr.avg)

class CIFAR10Evaluator(BaseEvaluator):
    def _init_loaders(self):
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.val_dataset = datasets.CIFAR10(
                root='./', download=True, train=False,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,]))
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=8, pin_memory=True)

class ImagenetEvaluator(BaseEvaluator):
    def _init_loaders(self):
        valdir = self.dataset_path
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.val_dataset = StridedImageFolder(
                valdir,
                transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,]),
                stride=self.stride)
        self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                sampler=self.val_sampler, num_workers=1, pin_memory=True,
                shuffle=False)

class ImagenetCEvaluator(BaseEvaluator):
    def __init__(self, corruption_type=None, corruption_name=None, corruption_level=None, **kwargs):
        self.corruption_type = corruption_type
        self.corruption_name = corruption_name
        self.corruption_level = corruption_level
        super().__init__(**kwargs)
    
    def _init_loaders(self):
        valdir = os.path.join(self.dataset_path, 'imagenet-c',
                              self.corruption_type, self.corruption_name, self.corruption_level)
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.val_dataset = StridedImageFolder(
                valdir,
                transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,]),
                stride=self.stride)
        self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                sampler=self.val_sampler, num_workers=1, pin_memory=True,
                shuffle=False)

class CIFAR10CEvaluator(BaseEvaluator):
    def __init__(self, corruption_type=None, corruption_name=None, corruption_level=None, **kwargs):
        self.corruption_type = corruption_type
        self.corruption_name = corruption_name
        self.corruption_level = corruption_level
        super().__init__(**kwargs)
    
    def _init_loaders(self):
        valdir = os.path.join(self.dataset_path, 'CIFAR-10-C')
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        self.val_dataset = CIFAR10C(valdir, transform=transform,
                                    corruption_name=self.corruption_name,
                                    corruption_level=self.corruption_level)
        self.val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                sampler=self.val_sampler, num_workers=1, pin_memory=True,
                shuffle=False)
