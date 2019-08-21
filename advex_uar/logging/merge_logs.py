import json

import click

@click.command()
@click.argument("files", nargs=-1)
@click.option("--out_file", type=click.Path())
def main(files, out_file=None):
    evals = []
    for filestr in files:
        with open(filestr, 'r') as f:
            x = json.load(f)
            evals = evals + x

    with open(out_file, 'w') as f:
        json.dump(evals, f)
              
if __name__ == '__main__':
    main()
