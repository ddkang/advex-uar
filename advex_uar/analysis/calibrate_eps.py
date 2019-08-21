import json

import click
import scipy.optimize

@click.command()
@click.option("--linf_ata_file", type=click.Path(exists=True), default='pgd_linf.out')
@click.option("--ata_file", type=click.Path(exists=True))
@click.option("--out_file", type=click.Path())
@click.option("--attack", default=None)
@click.option("--max_eps", default=-1.0)
@click.option("--table/--no_table", default=False)
@click.option("--resol", default=224)
def main(linf_ata_file=None, ata_file=None, out_file=None, attack=None, max_eps=None, table=None,
         resol=None):
    with open(linf_ata_file, 'r') as f:
        linf_atas = json.load(f)
    with open(ata_file, 'r') as f:
        atas = json.load(f)
    linf_ata_list = [x[1] for x in linf_atas]
    attack_atas = [x for x in atas if x[0][0] == attack and (max_eps < 0 or x[0][1] <= max_eps)]

    cost_matrix = []
    for linf_idx, linf_ata in enumerate(linf_ata_list):
        cost_matrix.append([])
        for attack_idx, attack_ata in enumerate(attack_atas):
            cost_matrix[-1].append(abs(linf_ata - attack_ata[1]))
    _, selected = scipy.optimize.linear_sum_assignment(cost_matrix)
    selected_atas = sorted([attack_atas[x.item()] for x in selected], key=lambda x: x[0][1])
    with open(out_file, 'w') as f:
        json.dump(selected_atas, f)
    print('Selected eps for attack {} using max_eps = {:.3f}'.format(attack, max_eps))
    for attack, ata, _ in selected_atas:
        print('Attack: {:10} Eps: {:10.3f} Steps: {:3} Step Size: {:8.3f}: ATA: {:.3f}'.format(*attack, ata))
    if table:
        eps_str = '|'.join(map(lambda x: '{:.3f}'.format(x[0][1]), selected_atas))
        if selected_atas[0][0][0] == 'fw_l1':
            eps_str = '|'.join(map(lambda x: '{:.3f}'.format(x[0][1] * resol * resol * 3), selected_atas))
        print('|' + eps_str + '|')
        ata_str = '|'.join(map(lambda x: '{:.1f}'.format(x[1] * 100), selected_atas))
        print('|' + ata_str + '|')
        
if __name__ == '__main__':
    main()
