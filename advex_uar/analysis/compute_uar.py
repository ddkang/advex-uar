import json

import click

from advex_uar.analysis.compute_ata import parse_logs, get_attack,\
    get_attacks, get_defenses, get_defense_from_run_id

def get_defense_run_ids(eval_logs):
    run_ids = []
    for log in eval_logs:
        run_id = log['wandb_run_id']
        if run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids

def get_attack_types(atas):
    attack_types = {}
    for ata in atas:
        attack_type = ata[0][0]
        if attack_type not in attack_types:
            attack_types[attack_type] = [ata]
        else:
            attack_types[attack_type].append(ata)
    return attack_types

def compute_uar(run_id, atas, eval_logs, defense):
    logs = [log for log in eval_logs if log['wandb_run_id'] == run_id]
    def_attack = defense[0]
    epsilon = defense[1]
    attack_types = get_attack_types(atas)
    uar_scores = []
    for attack_type, atas in attack_types.items():
        model_acc = 0
        log_count = 0
        eps_found = []
        eps_vals = [x[0][1] for x in atas]
        for log in logs:
            if (log['attack'] == attack_type and
                min(map(lambda x: abs(x - log['epsilon']), eps_vals)) < 0.01 and
                log['epsilon'] not in eps_found):
                model_acc += log['adv_acc']
                log_count = log_count + 1
                eps_found.append(log['epsilon'])
        if log_count != 6:
            print('Have {} eval runs for {} ({:9} eps {:.3f}) with attack {:9} instead of 6; saw {}, need {}'
                  .format(log_count, run_id, def_attack, epsilon, attack_type, str(sorted(eps_found)),
                          str(eps_vals)))
        else:
            score = model_acc / sum([x[1] for x in atas])
            uar_scores.append((attack_type, score))
    return uar_scores

@click.command()
@click.option("--eval_log_file", type=click.Path(exists=True))
@click.option("--calibrated_eps_file", type=click.Path(exists=True))
@click.option("--train_log_file", type=click.Path(exists=True))
@click.option("--out_file", type=click.Path())
@click.option("--run_id", default=None, help="Training run ID to compute UAR for.  If"\
              "not specified, computes for all runs seen.")
@click.option("--max_eps_file", type=click.Path(exists=True))
def main(eval_log_file=None, train_log_file=None, calibrated_eps_file=None, out_file=None,
         run_id=None, max_eps_file=None):
    eval_logs = parse_logs(eval_log_file)
    train_logs = parse_logs(train_log_file)
    if max_eps_file is not None:
        with open(max_eps_file, 'r') as f:
            max_eps = json.load(f)
    defenses = get_defenses(train_logs)
    if run_id:
        defense_run_ids = [run_id]
    else:
        defense_run_ids = get_defense_run_ids(eval_logs)
    with open(calibrated_eps_file, 'r') as f:
        atas = json.load(f)
    all_scores = []
    for run_id in defense_run_ids:
        defense = get_defense_from_run_id(run_id, defenses)
        if max_eps_file is not None and defense[0] not in max_eps:
            print('Adversarially training against {} not found in max_eps, omitting'.format(defense[0]))
        elif max_eps_file is None or defense[1] <= max_eps[defense[0]]:
            uar_scores = compute_uar(run_id, atas, eval_logs, defense)
            all_scores.append((run_id, defense[0], defense[1], uar_scores))
            defense = get_defense_from_run_id(run_id, defenses)
            print('{:8} {:9} Eps: {:.3f}: {}'.format(run_id, defense[0], defense[1], [(x[0], '{:.3f}'.format(x[1])) for x in uar_scores]))

    with open(out_file, 'w') as f:
        json.dump(all_scores, f)
        
if __name__ == '__main__':
    main()
