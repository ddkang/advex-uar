import json

import click

def parse_logs(log_file):
    with open(log_file, 'r') as f:
        data = json.load(f)
    if type(data) == list:
        return data
    else:
        return [data]

def get_attack(log):
    return log['attack'], log['epsilon'], log['n_iters'], log['step_size']

def get_attacks(all_logs):
    attacks = []
    for log in all_logs:
        attack = get_attack(log)
        if attack not in attacks:
            attacks.append(attack)
    return attacks

def get_defense(log):
    std_acc = log['val_std_acc'] / 100
    return log['attack'], log['epsilon'], log['n_iters'],\
        log['step_size'], log['run_id'], log['adv_train'],\
        std_acc

def get_defenses(train_logs):
    defenses = []
    for log in train_logs:
        defense = get_defense(log)
        if defense not in defenses:
            defenses.append(defense)
    return defenses

def get_defense_from_run_id(run_id, defenses):
    for defense in defenses:
        if defense[4] == run_id:
            return defense
    return None

def is_type_match(attack, defense):
    if defense[0] is None:
        return False
    else:
        return (attack[0] == defense[0])

def get_ata_val(attack, eval_logs, defenses):
    # (-1, 'none') is a sentinel value
    adv_acc_vals = [(-1, 'none')]
    for log in eval_logs:
        if attack == get_attack(log):
            run_id = log['wandb_run_id']
            defense = get_defense_from_run_id(run_id, defenses)
            if is_type_match(attack, defense):
                adv_acc_vals.append((log['adv_acc'], run_id))
    return max(adv_acc_vals, key=lambda x: x[0])

@click.command()
@click.option("--eval_log_file", type=click.Path(exists=True))
@click.option("--train_log_file", type=click.Path(exists=True))
@click.option("--out_file", type=click.Path())
def main(eval_log_file=None, train_log_file=None, out_file=None):
    eval_logs = parse_logs(eval_log_file)
    train_logs = parse_logs(train_log_file)
    attacks = get_attacks(eval_logs)
    defenses = get_defenses(train_logs)
    attacks = sorted(attacks, key=lambda x: (x[0], x[1]))
    ata_vals = []
    for attack in attacks:
        ata, run_id = get_ata_val(attack, eval_logs, defenses)
        ata_vals.append((attack, ata, run_id))
    with open(out_file, 'w') as f:
        json.dump(ata_vals, f)
    for attack, ata, run_id in ata_vals:
        print('Attack: {:10} Eps: {:10.3f} Steps: {:3} Step Size: {:8.3f}: ATA: {:6.3f} Run ID: {}'.format(*attack, ata, run_id))
        
if __name__ == '__main__':
    main()
