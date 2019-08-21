import importlib
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms, models
import horovod.torch as hvd

from advex_uar.common.loader import StridedImageFolder

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def correct(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float()

class BaseTrainer():
    # Notes:
    # The attack needs to be initialized after the cuda device is set
    def __init__(self, **kwargs):
        default_attr = dict(
            # Training options
            batch_size=32, base_lr=0.0125, momentum=0.9, wd=1e-4, epochs=90, warmup_epochs=5,
            stride=10, label_smoothing=-1.0, rand_target=False,
            # validation options
            run_val=True,
            # Model/checkpoint options
            model=None, checkpoint_dir=None, dataset_path='/mnt/imagenet-test/',
            # Attack options
            attack=None, attack_backward_steps=0, attack_loss='avg', scale_eps=False, rand_init=True,
            # Communication options
            fp16_allreduce=False,
            # Logging options
            logger=None)
        default_attr.update(kwargs)
        for k in default_attr:
            setattr(self, k, default_attr[k])
        assert self.attack_loss in ['avg', 'adv_only', 'logsumexp', 'max']

        # Validate args
        assert self.model != None

        # Set up checkpointing
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cuda = True
        self.batches_per_allreduce = 1
        self.verbose = 1 if hvd.rank() == 0 else 0
        self.compression = hvd.Compression.fp16 if self.fp16_allreduce else hvd.Compression.none

        if self.verbose:
            print(self.model)

        torch.cuda.set_device(hvd.local_rank())

        if self.cuda:
            self.model.cuda()

        if self.attack:
            self.attack = self.attack()
            self.attack_backward_steps = self.attack.nb_backward_steps

        self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._init_loaders()
        self._init_optimizer()
        self._start_sync()

    def _init_loaders(self):
        raise NotImplementedError

    def _init_optimizer(self):
        optimizer = optim.SGD(
                self.model.parameters(),
                lr=(self.base_lr * self.batches_per_allreduce * hvd.size()),
                momentum=self.momentum, weight_decay=self.wd)

        steps_per_batch = self.attack_backward_steps + 1 + 100000000
        if self.attack_loss == 'avg':
            steps_per_batch += 1
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(),
            compression=self.compression,
            backward_passes_per_step=steps_per_batch * self.batches_per_allreduce)

    def _start_sync(self):
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

    def _adjust_learning_rate(self, epoch, batch_idx):
        if epoch < self.warmup_epochs:
            epoch += float(batch_idx + 1) / len(self.train_loader)
            lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / self.warmup_epochs + 1)
        elif epoch < 30:
            lr_adj = 1.
        elif epoch < 60:
            lr_adj = 1e-1
        elif epoch < 80:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * hvd.size() * self.batches_per_allreduce * lr_adj

    def _checkpoint(self, epoch):
        if self.checkpoint_dir and hvd.local_rank() == 0:
            out_fname = '{:02d}.pth'.format(epoch)
            out_fname = os.path.join(self.checkpoint_dir, out_fname)
            state = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
            torch.save(state, out_fname)

    def _compute_loss(self, output, target):
        if self.label_smoothing > 0:
            n_class = len(self.val_dataset.classes)
            one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot.clamp(self.label_smoothing / (n_class - 1), 1 - self.label_smoothing)
            log_prob = F.log_softmax(output, dim=1)
            loss = -(one_hot * log_prob).sum(dim=1)
            return loss.mean()
        else:
            return F.cross_entropy(output, target)
        
    def _train_epoch(self, epoch):
        self.model.train()

        train_std_loss = Metric('train_std_loss')
        train_std_acc = Metric('train_std_acc')
        train_adv_loss = Metric('train_adv_loss')
        train_adv_acc = Metric('train_adv_acc')

        if self.attack:
            self.attack.set_epoch(epoch)
            
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            self._adjust_learning_rate(epoch, batch_idx)
            loss = torch.zeros([], dtype=torch.float32, device='cuda')
            if (not self.attack) or self.attack_loss == 'avg':
                output = self.model(data)
                loss += self._compute_loss(output, target)
                train_std_loss.update(loss)
                train_std_acc.update(accuracy(output, target))
            else:
                with torch.no_grad():
                    self.model.eval()
                    output = self.model(data)
                    train_std_loss_val = self._compute_loss(output, target)            
                    train_std_loss.update(train_std_loss_val)
                    train_std_acc.update(accuracy(output, target))
                    self.model.train()

            if self.attack:
                if self.rand_target:
                    attack_target = torch.randint(
                        0, len(self.val_dataset.classes) - 1, target.size(),
                        dtype=target.dtype, device='cuda')
                    attack_target = torch.remainder(target + attack_target + 1, len(self.val_dataset.classes))
                adv_loss = torch.zeros([], dtype=torch.float32, device='cuda')
                if self.rand_target:
                    data_adv = self.attack(self.model, data, attack_target,
                                           avoid_target=False, scale_eps=self.scale_eps)
                else:
                    data_adv = self.attack(self.model, data, target,
                                           avoid_target=True, scale_eps=self.scale_eps)
                output_adv = self.model(data_adv)
                adv_loss = self._compute_loss(output_adv, target)
                
                train_adv_loss.update(adv_loss)
                train_adv_acc.update(accuracy(output_adv, target))
                loss += adv_loss
                if self.attack_loss == 'avg':
                    loss /= 2.

            self.optimizer.synchronize()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if hvd.rank() == 0:
            log_dict = {'train_std_loss':train_std_loss.avg.item(),
                        'train_std_acc':train_std_acc.avg.item(),
                        'train_adv_loss':train_adv_loss.avg.item(),
                        'train_adv_acc':train_adv_acc.avg.item()}
            print(log_dict)
            self.logger.log(log_dict, epoch)

    def _val_epoch(self, epoch):
        self.model.eval()

        val_std_loss = Metric('val_std_loss')
        val_std_acc = Metric('val_std_acc')

        val_adv_acc = Metric('val_adv_acc')
        val_adv_loss = Metric('val_adv_loss')
        val_max_adv_acc = Metric('val_max_adv_acc')
        val_max_adv_loss = Metric('val_max_adv_loss')

        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(data)
                val_std_loss.update(F.cross_entropy(output, target))
                val_std_acc.update(accuracy(output, target))
            if self.attack:
                rand_target = torch.randint(
                    0, len(self.val_dataset.classes) - 1, target.size(),
                    dtype=target.dtype, device='cuda')
                rand_target = torch.remainder(target + rand_target + 1, len(self.val_dataset.classes))
                data_adv = self.attack(self.model, data, rand_target,
                                       avoid_target=False, scale_eps=self.scale_eps)
                data_max_adv = self.attack(self.model, data, rand_target, avoid_target=False, scale_eps=False)
                with torch.no_grad():
                    output_adv = self.model(data_adv)
                    val_adv_loss.update(F.cross_entropy(output_adv, target))
                    val_adv_acc.update(accuracy(output_adv, target))
                    
                    output_max_adv = self.model(data_max_adv)
                    val_max_adv_loss.update(F.cross_entropy(output_max_adv, target))
                    val_max_adv_acc.update(accuracy(output_max_adv, target))
            self.model.eval()

        if hvd.rank() == 0:
            log_dict = {'val_std_loss':val_std_loss.avg.item(),
                        'val_std_acc':val_std_acc.avg.item(),
                        'val_adv_loss':val_adv_loss.avg.item(),
                        'val_adv_acc':val_adv_acc.avg.item(),
                        'val_adv_loss':val_max_adv_loss.avg.item(),
                        'val_max_adv_acc':val_max_adv_acc.avg.item()}
            self.logger.log(log_dict, epoch)

        if self.verbose:
            print(log_dict)

        self.optimizer.synchronize()
        self.optimizer.zero_grad()

    def train(self):
        if hvd.rank() == 0:
            print('Beginning training with {} epochs'.format(self.epochs))
        for epoch in range(self.epochs):
            begin = time.time()
            self._train_epoch(epoch)
            if self.run_val:
                self._val_epoch(epoch)
            self._checkpoint(epoch)
            end = time.time()
            if self.verbose:
                print('Epoch {} took {:.2f} seconds'.format(epoch, end - begin))
        if hvd.rank() == 0:
            self.logger.log_ckpt(self.model, self.optimizer)
            self.logger.end(summarize_vals=True)

class ImagenetTrainer(BaseTrainer):
    def _init_loaders(self):
        allreduce_batch_size = self.batch_size * self.batches_per_allreduce

        traindir = os.path.join(self.dataset_path, 'train')
        valdir = os.path.join(self.dataset_path, 'val')
        self.train_dataset = StridedImageFolder(
                traindir,
                transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        self.normalize,]),
                stride=self.stride)
        self.val_dataset = StridedImageFolder(
                valdir,
                transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        self.normalize,]),
                stride=self.stride)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=allreduce_batch_size,
                sampler=self.train_sampler, num_workers=8, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=allreduce_batch_size,
                sampler=self.val_sampler, num_workers=8, pin_memory=True,
                shuffle=False)

class CIFAR10Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        if 'epochs' not in kwargs:
            kwargs['epochs'] = 200
        if 'base_lr' not in kwargs:
            kwargs['base_lr'] = 0.1
        super().__init__(**kwargs)

    def _adjust_learning_rate(self, epoch, batch_idx):
        if epoch < 100:
            lr = 0.1
        elif epoch < 150:
            lr = 0.01
        else:
            lr = 0.001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _init_loaders(self):
        allreduce_batch_size = self.batch_size * self.batches_per_allreduce

        if hvd.local_rank() != 0:
            hvd.allreduce(torch.tensor(0), name='barrier')
        self.train_dataset = datasets.CIFAR10(
                root=self.dataset_path, download=(hvd.local_rank() == 0),
                train=True,
                transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        self.normalize,]))
        if hvd.local_rank() == 0:
            hvd.allreduce(torch.tensor(0), name='barrier')
        self.val_dataset = datasets.CIFAR10(
                root=self.dataset_path,
                train=False,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        self.normalize,]))
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=allreduce_batch_size,
                shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=allreduce_batch_size,
                shuffle=False, num_workers=8, pin_memory=True)
