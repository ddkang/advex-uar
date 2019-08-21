import json
import time

import click
import wandb

EVAL_CONFIG_KEYS = ["class_downsample_factor", "attack", "use_wandb",
                    "ckpt_path", "dataset", "resnet_size", "batch_size",
                    "n_iters", "step_size", "epsilon", "wandb_username",
                    "wandb_ckpt_project", "wandb_project", "wandb_run_id",
                    "tag", "use_fp16", "corruption_type", "dataset_path", 
                    "corruption_name", "corruption_level", "use_max_step",
                    'run_id']
EVAL_SUMMARY_KEYS = ["std_acc", "adv_acc"]

TRAIN_CONFIG_KEYS = ["class_downsample_factor", "attack", "dataset",
                     "resnet_size", "use_wandb", "epochs", "wandb_project",
                     "tag", "batch_size", "label_smoothing", "dataset_path",
                     "checkpoint_dir", "use_fp16", "adv_train", "attack_loss",
                     "rand_target", "epsilon", "step_size", "use_max_step",
                     "n_iters", "scale_each", "scale_eps", "run_id"]
TRAIN_SUMMARY_KEYS = ["train_std_loss", "train_std_acc", "train_adv_loss",
                      "train_adv_acc", "val_std_loss", "val_std_acc",
                      "val_adv_loss", "val_adv_acc", "val_max_adv_acc"]

def extract_summary(run):
    summary_dict = {}
    if 'adv_train' in run.config:
        config_keys, summary_keys = TRAIN_CONFIG_KEYS, TRAIN_SUMMARY_KEYS
    else:
        config_keys, summary_keys = EVAL_CONFIG_KEYS, EVAL_SUMMARY_KEYS
    for k in run.config:
        summary_dict[k] = run.config[k]

    if 'adv_train' in run.config and type(run.config['adv_train']) == dict and not run.config['adv_train']['value']:
        summary_dict['attack'] = 'clean'
        summary_dict['epsilon'] = 0
        summary_dict['n_iters'] = 0
        summary_dict['step_size'] = 0
    if 'train_many' in run.config and type(run.config['train_many']) != dict and run.config['train_many']:
        summary_dict['train_many'] = True
        summary_dict['attack'] = run.config['attack_strs']
        
    for k in summary_keys:
        if k in run.summary._json_dict:
            # Need to do this because run.summary has lazy loading
            summary_dict[k] = run.summary[k]
        else:
            run_id = run.config['run_id']
            print('Run {} is missing {}'.format(run_id, k))
    return summary_dict

def dump_single_run(run, output_file):
    summary = extract_summary(run)
    with open(output_file, 'w') as f:
        json.dump(summary, f)

def dump_many_runs(run_gen, output_file):
    summaries = []
    count = 0
    for run in run_gen:
        summaries.append(extract_summary(run))
        count = count + 1
        if count % 100 == 0:
            print('Downloaded {:4} runs'.format(count))
    with open(output_file, 'w') as f:
        json.dump(summaries, f)

@click.command()
# wandb options
@click.option("--wandb_username", help="Username for WandB")
@click.option("--wandb_project", help="Project for WandB")
@click.option("--run_id", default=None, help="WandB run ID to get logs for.  If"\
              "not specified, downloads all runs in a project")
@click.option("--output_file", help="Filename (with path) to output log to")
def main(wandb_username=None, wandb_project=None, run_id=None, output_file=None):
    api = wandb.Api()
    if run_id:
        run = api.run('{}/{}/{}'.format(wandb_username, wandb_project, run_id))
        dump_single_run(run, output_file)
    else:
        run_gen = api.runs('{}/{}'.format(wandb_username, wandb_project),
                           {'$and': [{'state':'finished'}]})
        dump_many_runs(run_gen, output_file)
        
if __name__ == '__main__':
    main()
