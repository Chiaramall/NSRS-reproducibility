"""Shared validation/final protocol, also compatible with KGAT's Python 3.7."""
import json
import math
import os
from pathlib import Path


class RunProtocol:
    def __init__(self, phase=None, result_file=None):
        self.phase = phase or os.environ.get('NSRS_PHASE', 'legacy')
        if self.phase not in ('legacy', 'validation', 'final'):
            raise ValueError('Unknown NSRS_PHASE: ' + self.phase)
        destination = result_file or os.environ.get('NSRS_RESULT_FILE')
        self.result_file = Path(destination) if destination else None
        if self.phase != 'legacy' and self.result_file is None:
            raise ValueError('NSRS_RESULT_FILE is required for controlled runs')
        self.best = None
        self.evaluated_epochs = []

    @property
    def controlled(self):
        return self.phase != 'legacy'

    def should_evaluate(self, epoch, budget, legacy_condition=True):
        if self.phase == 'legacy':
            return legacy_condition
        # One-based epoch counts are also the final retraining budgets.
        return self.phase == 'validation' or epoch == budget

    def observe(self, epoch, metrics):
        if not self.controlled:
            return
        def scalar(name):
            value = metrics.get(name + '@20', metrics.get(name))
            try:
                value = value[0]
            except (TypeError, IndexError):
                pass
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid metric %s: %s' % (name, value))
            return value
        current = {'recall@20': scalar('recall'), 'ndcg@20': scalar('ndcg'),
                   'selected_epoch': int(epoch)}
        self.evaluated_epochs.append(int(epoch))
        if self.phase == 'final' and len(self.evaluated_epochs) != 1:
            raise RuntimeError('Final test must be evaluated exactly once')
        # Keep metrics from one epoch; break ties by NDCG, then earliest epoch.
        if self.best is None or (current['recall@20'], current['ndcg@20']) > (self.best['recall@20'], self.best['ndcg@20']):
            self.best = current
        result = dict(self.best, phase=self.phase,
                      evaluated_epochs=self.evaluated_epochs, last_epoch=int(epoch))
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.result_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(self.result_file)


protocol = RunProtocol()
