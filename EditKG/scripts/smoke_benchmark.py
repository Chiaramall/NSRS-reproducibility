#!/usr/bin/env python3
"""Exercise the complete protocol on a small synthetic dataset (not paper scores)."""
import argparse
from pathlib import Path
import subprocess
import sys

from prepare_validation_splits import create_split


def prepare(root):
    source = root / 'full/last-fm'
    source.mkdir(parents=True, exist_ok=True)
    train, test = [], []
    for user in range(64):
        items = [(user * 7 + offset) % 80 for offset in range(18)]
        train.append(' '.join(map(str, [user] + items[:16])))
        test.append(' '.join(map(str, [user] + items[16:])))
    (source / 'train.txt').write_text('\n'.join(train) + '\n')
    (source / 'test.txt').write_text('\n'.join(test) + '\n')
    (source / 'kg_final.txt').write_text(''.join('%s %s %s\n' % (item, offset % 4, 80 + (item + offset) % 32)
                                               for item in range(80) for offset in range(8)))
    for name, count in [('item', 80), ('entity', 112), ('relation', 4), ('user', 64)]:
        (source / (name + '_list.txt')).write_text('org_id remap_id\n' + ''.join('%d %d\n' % (index, index) for index in range(count)))
    create_split(source, root / 'validation/last-fm', 'last-fm', 0.2, 2024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['all'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--kgat-python')
    args = parser.parse_args()
    root = args.output.resolve()
    prepare(root)
    command = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
               '--models', *args.models, '--datasets', 'last-fm',
               '--data-root', str(root / 'full'), '--validation-root', str(root / 'validation'),
               '--output', str(root / 'results'), '--epochs', '2', '--trials', '2',
               '--final-seeds', '2024']
    if args.cpu: command.append('--cpu')
    if args.kgat_python: command.extend(['--kgat-python', args.kgat_python])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
