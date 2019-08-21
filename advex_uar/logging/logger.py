import importlib
import json
import os
import subprocess

import torch

def configure_wandb(use_wandb, proj_name):
    if os.getenv('WANDB_API_KEY') is None or not use_wandb:
        os.environ['WANDB_MODE'] = 'dryrun'
    else:
        os.environ['WANDB_PROJECT'] = proj_name

def init_wandb(job_type, flag_dict):
    out = wandb.init(job_type=job_type)
    print('wandb inited', out, wandb.config)
    for k, v in flag_dict.items():
        setattr(wandb.config, k, v)
    setattr(wandb.config, 'run_id', wandb.run.id)

class Logger(object):
    def __init__(self, use_wandb, job_type, run_id, log_dir=None, flags=None):
        self.use_wandb = use_wandb
        self.job_type = job_type
        self.log_dir = log_dir
        self.flags = flags

        if self.use_wandb:
            globals()['wandb'] = importlib.import_module('wandb')
            configure_wandb(use_wandb, flags['wandb_project'])
            init_wandb(job_type, flags)
            self.log_dir = wandb.run.dir
            run_id = wandb.run.id # FIXME: hack

        if job_type == 'eval':
            self.img_dir = os.path.join(self.log_dir, 'images')
            if self.img_dir is not None:
                os.mkdir(self.img_dir)
        self.vals = {}
        self.summary = {}
        if not self.use_wandb:
            self.log_summary(flags)
            if self.job_type == 'train':
                self.log_summary({'run_id': run_id})

    def log(self, log_dict, step):
        if self.use_wandb:
            wandb.log(log_dict, step=step)
        else:
            for k, v in log_dict.items():
                if k not in self.vals:
                    self.vals[k] = []
                self.vals[k].append((v, step))

    def log_image(self, image, filestr):
        img_file = os.path.join(self.img_dir, filestr)
        image.save(img_file)
                
    def log_summary(self, log_dict):
        if self.use_wandb:
            for k, v in log_dict.items():
                wandb.run.summary[k] = v
        else:
            for k, v in log_dict.items():
                self.summary[k] = v

    def log_ckpt(self, model, optim):
        state = {'model': model.state_dict(),
                 'optimizer': optim.state_dict()}
        # automatically uploaded to wandb after the run
        torch.save(state, os.path.join(self.log_dir, 'ckpt.pth'))
                
    def write(self):
        if not self.use_wandb:
            all_file = os.path.join(self.log_dir, 'all.log')
            with open(all_file, 'w') as f:
                json.dump(self.vals, f)

    def write_summary(self):
        if not self.use_wandb:
            summary_file = os.path.join(self.log_dir, 'summary.log')
            with open(summary_file, 'w') as f:
                json.dump(self.summary, f)
            
    def end(self, summarize_vals=False):
        if self.use_wandb:
            zip_cmd = subprocess.Popen(['zip', '-r', os.path.join(wandb.run.dir, 'images.zip'),
                                        self.img_dir],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            result = zip_cmd.communicate()
            print(result)
            rm_cmd = subprocess.Popen(['rm', '-rf', self.img_dir])
            result = rm_cmd.communicate()
            print(result)
        else:
            if summarize_vals:
                summary_dict = {}
                for k, v in self.vals.items():
                    summary_dict[k] = v[-1][0]
                self.log_summary(summary_dict)
            self.write_summary()
