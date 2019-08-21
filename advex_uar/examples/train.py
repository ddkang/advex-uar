import click

import horovod.torch as hvd
import torch
import numpy as np

from advex_uar.train import ImagenetTrainer, CIFAR10Trainer
from advex_uar.common.pyt_common import *
from advex_uar.common import FlagHolder

def train(**flag_kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**flag_kwargs)
    hvd.init()
    if FLAGS.step_size is None:
        FLAGS.step_size = get_step_size(FLAGS.epsilon, FLAGS.n_iters, FLAGS.use_max_step)
        FLAGS._dict['step_size'] = FLAGS.step_size

    if hvd.rank() == 0:
        FLAGS.summary()

    if FLAGS.dataset == 'imagenet':
        Trainer = ImagenetTrainer
    elif FLAGS.dataset == 'cifar-10':
        Trainer = CIFAR10Trainer
    else:
        raise NotImplementedError

    if hvd.rank() == 0:
        logger = init_logger(FLAGS.use_wandb, 'train', FLAGS._dict)
    else:
        logger = None
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = logger.log_dir
    print('checkpoint at {}'.format(FLAGS.checkpoint_dir))

    model = get_model(FLAGS.dataset, FLAGS.resnet_size, 1000 // FLAGS.class_downsample_factor)
    if FLAGS.adv_train:
        attack = get_attack(FLAGS.dataset, FLAGS.attack, FLAGS.epsilon,
                            FLAGS.n_iters, FLAGS.step_size, FLAGS.scale_each)
    else:
        attack = None

    trainer = Trainer(
        # model/checkpoint options
        model=model, checkpoint_dir=FLAGS.checkpoint_dir, dataset_path=FLAGS.dataset_path,
        # attack options
        attack=attack, scale_eps=FLAGS.scale_eps, attack_loss=FLAGS.attack_loss,
        # training options
        batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, stride=FLAGS.class_downsample_factor,
        fp_all_reduce=FLAGS.use_fp16, label_smoothing=FLAGS.label_smoothing, rand_target=FLAGS.rand_target,
        # logging options
        logger=logger, tag=FLAGS.tag)
    trainer.train()

    if hvd.rank() == 0:
        print("Training finished.")

@click.command()
# wandb options
@click.option("--use_wandb/--no_wandb", is_flag=True, default=True)
@click.option('--wandb_project', default='advex_uar_train', help="WandB project to log to")
@click.option("--tag", default='train', help="Short tag for WandB")

# Dataset options
# Allowed values: ['imagenet', 'cifar-10']
@click.option('--dataset', default='imagenet')
@click.option("--dataset_path", default='.', help="Location of the training data")

# Model options
@click.option("--resnet_size", default=50)
@click.option("--class_downsample_factor", default=1, type=int)

# Training options
@click.option("--batch_size", default=32)
@click.option("--epochs", default=90)
@click.option("--label_smoothing", default=0.0)
@click.option("--checkpoint_dir", default=None, "Location to write the final ckpt to")
@click.option("--use_fp16/--no_fp16", is_flag=True, default=False)

# Adversarial training options
@click.option("--adv_train/--no_adv_train", is_flag=True, default=False)
@click.option("--attack_loss", default='adv_only') # 'avg', 'adv_only', or 'logsumexp'
@click.option("--rand_target/--no_rand_target", is_flag=True, default=True)

# Attack options
# Allowed values: ['pgd_linf', 'pgd_l2', 'fw_l1', 'jpeg_linf', 'jpeg_l2', 'jpeg_l1', 'elastic', 'fog', 'gabor', 'snow']
@click.option("--attack", default=None, type=str)
@click.option("--epsilon", default=16.0, type=float)
@click.option("--step_size", default=None, type=float)
@click.option("--use_max_step", is_flag=True, default=False)
@click.option("--n_iters", default=10, type=int)
@click.option("--scale_each/--no_scale_each", is_flag=True, default=True)
@click.option("--scale_eps/--no_scale_eps", is_flag=True, default=True)

def main(**flags):
    train(**flags)

if __name__ == '__main__':
    main()
