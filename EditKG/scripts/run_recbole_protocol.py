#!/usr/bin/env python3
"""RecBole training with a full validation split or a fixed final epoch budget."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/external/recbole'))
sys.path.append(str(ROOT))
from nsrs_protocol import protocol
from prepare_baseline_data import read_users, write_inter


def prepare(source, destination, dataset):
    destination.mkdir(parents=True, exist_ok=True)
    train = list(read_users(source / 'train.txt'))
    evaluation = list(read_users(source / 'test.txt'))
    write_inter(destination / (dataset + '.train.inter'), train)
    write_inter(destination / (dataset + '.valid.inter'), evaluation if protocol.phase == 'validation' else [])
    write_inter(destination / (dataset + '.test.inter'), evaluation if protocol.phase == 'final' else [])
    with (destination / (dataset + '.kg')).open('w') as out, (source / 'kg_final.txt').open() as inp:
        out.write('head_id:token\trelation_id:token\ttail_id:token\n')
        for line in inp:
            head, relation, tail = line.split()
            out.write('e%s\tr%s\te%s\n' % (head, relation, tail))
    with (destination / (dataset + '.link')).open('w') as out:
        out.write('item_id:token\tentity_id:token\n')
        for item in range(max(int(item) for _, items in train + evaluation for item in items) + 1):
            out.write('%s\te%s\n' % (item, item))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if not protocol.controlled:
        parser.error('NSRS_PHASE must be validation or final')
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed

    atomic = args.work_dir / 'atomic'
    prepare(args.data_root / args.dataset, atomic / args.dataset, args.dataset)
    overrides = json.loads(args.params.read_text())
    if 'reg' in overrides:
        overrides['reg_weights'] = [overrides.pop('reg')] * 2
    overrides.update(data_path=str(atomic), epochs=args.epochs, seed=args.seed,
                     eval_step=1, stopping_step=args.epochs + 1,
                     checkpoint_dir=str(args.work_dir / 'checkpoints'),
                     use_gpu=not args.cpu, gpu_id='0', show_progress=False,
                     save_dataset=False, save_dataloaders=False,
                     metrics=['Recall', 'NDCG'], topk=[20], valid_metric='Recall@20')
    config = Config(model=args.model.upper(), dataset=args.dataset,
                    config_file_list=[str(ROOT / 'configs/recbole_lastfm.yaml')],
                    config_dict=overrides)
    init_seed(args.seed, True)
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train._dataset).to(config['device'])
    base = get_trainer(config['MODEL_TYPE'], config['model'])

    class ObservedTrainer(base):
        def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
            self.protocol_epoch = epoch_idx + 1
            return super()._train_epoch(train_data, epoch_idx, *args, **kwargs)

        def _valid_epoch(self, *args, **kwargs):
            score, metrics = super()._valid_epoch(*args, **kwargs)
            protocol.observe(self.protocol_epoch, metrics)
            return score, metrics

    trainer = ObservedTrainer(config, model)
    # No validation loader is passed during final retraining.
    trainer.fit(train, valid if protocol.phase == 'validation' else None,
                saved=False, show_progress=False)
    if protocol.phase == 'final':
        metrics = trainer.evaluate(test, load_best_model=False, show_progress=False)
        protocol.observe(args.epochs, metrics)
    trainer.tensorboard.close()


if __name__ == '__main__':
    main()
