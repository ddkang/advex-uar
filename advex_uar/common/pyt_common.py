import datetime
import functools
import math
import os
import uuid

from torchvision import models

from advex_uar.attacks import PGDAttack, ElasticAttack, FrankWolfeAttack,\
    JPEGAttack, GaborAttack, FogAttack, SnowAttack
from advex_uar.common.models import cifar10_resnet
from advex_uar.logging.logger import Logger

def init_logger(use_wandb, job_type, flags):
    if use_wandb:
        log_dir = None
        run_id = None
    else:
        dir_path = os.getcwd()
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        dir_str = '{}-{}-{}'.format(job_type, time_str, run_id)
        log_dir = os.path.join(dir_path, job_type, dir_str)
        os.makedirs(log_dir, exist_ok=True)
    logger = Logger(use_wandb, job_type, run_id, log_dir=log_dir, flags=flags)
    return logger
        
def get_imagenet_model(resnet_size, nb_classes):
    size_to_model = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152
    }
    return size_to_model[resnet_size](num_classes=nb_classes)

def get_cifar10_model(resnet_size):
    size_to_model = {
        20: cifar10_resnet.resnet20,
        32: cifar10_resnet.resnet32,
        44: cifar10_resnet.resnet44,
        56: cifar10_resnet.resnet56,
        110: cifar10_resnet.resnet110,
    }
    model = size_to_model[resnet_size]()
    return model

def get_model(dataset, resnet_size, nb_classes):
    if dataset in ['imagenet', 'imagenet-c']:
        return get_imagenet_model(resnet_size, nb_classes)
    elif dataset in ['cifar-10', 'cifar-10-c']:
        return get_cifar10_model(resnet_size)

def _get_attack(dataset, attack, eps, n_iters, step_size, scale_each):
    if dataset in ['imagenet', 'imagenet-c']:
        resol = 224
        elastic_kernel = 25
        elastic_std = 3
    elif dataset in ['cifar-10', 'cifar-10-c']:
        resol = 32
        elastic_kernel = 5
        elastic_std = 3.0/224.0 * 32

    if attack == 'pgd_linf':
        return functools.partial(PGDAttack, n_iters, eps,
                                 step_size, resol, norm='linf',
                                 scale_each=scale_each)
    elif attack == 'pgd_l2':
        return functools.partial(PGDAttack, n_iters, eps,
                                 step_size, resol, norm='l2',
                                 scale_each=scale_each)
    elif attack == 'fw_l1':
        return functools.partial(FrankWolfeAttack, n_iters, eps,
                                 resol, scale_each=scale_each)
    elif attack == 'jpeg_linf':
        return functools.partial(JPEGAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each,
                                 opt='linf')
    elif attack == 'jpeg_l2':
        return functools.partial(JPEGAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each,
                                 opt='l2')
    elif attack == 'jpeg_l1':
        return functools.partial(JPEGAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each,
                                 opt='l1')
    elif attack == 'elastic':
        return functools.partial(ElasticAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each,
                                 kernel_size=elastic_kernel,
                                 kernel_std=elastic_std)
    elif attack == 'fog':
        return functools.partial(FogAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each)
    elif attack == 'gabor':
        return functools.partial(GaborAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each)
    elif attack == 'snow':
        return functools.partial(SnowAttack, n_iters, eps,
                                 step_size, resol, scale_each=scale_each)
    else:
        raise NotImplementedError    
    
def get_attack(*args, **kwargs):
    return _get_attack(*args, **kwargs)

def get_step_size(epsilon, n_iters, use_max=False):
    if use_max:
        return epsilon
    else:
        return epsilon / math.sqrt(n_iters)
