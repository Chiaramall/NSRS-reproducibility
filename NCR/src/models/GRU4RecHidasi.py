# coding=utf-8

import ast
import math

import torch
from torch import nn
from torch.optim import Optimizer

from models.RecModel import RecModel
from utils.global_p import C_HISTORY, C_HISTORY_LENGTH, X


class IndexedAdagradM(Optimizer):
    """Adagrad with optional momentum and sparse embedding updates."""

    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum: {}'.format(momentum))
        if eps <= 0.0:
            raise ValueError('Invalid epsilon: {}'.format(eps))
        super(IndexedAdagradM, self).__init__(
            params, dict(lr=lr, momentum=momentum, eps=eps))
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                state['acc'] = torch.zeros_like(parameter)
                if momentum > 0.0:
                    state['mom'] = torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                lr = group['lr']
                momentum = group['momentum']
                eps = group['eps']
                if grad.is_sparse:
                    grad = grad.coalesce()
                    indices = grad.indices()[0]
                    values = grad.values()
                    accumulators = state['acc'][indices] + values.pow(2)
                    state['acc'].index_copy_(0, indices, accumulators)
                    denominators = accumulators.add(eps).sqrt()
                    if momentum > 0.0:
                        momenta = state['mom'][indices]
                        momenta.mul_(momentum).addcdiv_(
                            values, denominators, value=-lr)
                        state['mom'].index_copy_(0, indices, momenta)
                        parameter.index_add_(0, indices, momenta)
                    else:
                        parameter.index_add_(
                            0, indices, values / denominators, alpha=-lr)
                else:
                    state['acc'].add_(grad.pow(2))
                    denominators = state['acc'].add(eps).sqrt()
                    if momentum > 0.0:
                        state['mom'].mul_(momentum).addcdiv_(
                            grad, denominators, value=-lr)
                        parameter.add_(state['mom'])
                    else:
                        parameter.addcdiv_(grad, denominators, value=-lr)
        return loss


def init_parameter_matrix(tensor, dim0_scale=1, dim1_scale=1):
    sigma = math.sqrt(
        6.0 / float(tensor.size(0) / dim0_scale +
                    tensor.size(1) / dim1_scale))
    nn.init.uniform_(tensor, -sigma, sigma)


class GRU4RecHidasi(RecModel):
    """Hidasi GRU4Rec adapted to NCR's history-prefix batches.

    NCR supplies complete history prefixes rather than the official
    session-parallel iterator. Positive targets are used as in-batch negatives
    together with NCR's sampled negatives.
    """

    data_processor = 'ProLogicRecDP'
    uses_custom_initialization = True
    skip_gradient_clipping = True

    @staticmethod
    def parse_model_args(parser, model_name='GRU4RecHidasi'):
        parser.add_argument('--hidasi_layers', type=str, default='[64]',
                            help='GRU hidden sizes, e.g. "[64]".')
        parser.add_argument('--hidasi_loss', type=str, default='bpr-max',
                            choices=['bpr-max', 'cross-entropy'])
        parser.add_argument('--hidasi_embedding', type=int, default=0)
        parser.add_argument('--hidasi_constrained_embedding', type=int,
                            default=1, choices=[0, 1])
        parser.add_argument('--hidasi_dropout_embed', type=float, default=0.0)
        parser.add_argument('--hidasi_dropout_hidden', type=float, default=0.0)
        parser.add_argument('--hidasi_momentum', type=float, default=0.0)
        parser.add_argument('--hidasi_bpreg', type=float, default=1.0)
        parser.add_argument('--hidasi_elu_param', type=float, default=0.5)
        parser.add_argument('--hidasi_adagrad_eps', type=float, default=1e-6)
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, hidasi_layers, hidasi_loss, hidasi_embedding,
                 hidasi_constrained_embedding, hidasi_dropout_embed,
                 hidasi_dropout_hidden, hidasi_momentum, hidasi_bpreg,
                 hidasi_elu_param, hidasi_adagrad_eps, *args, **kwargs):
        self.layers = (hidasi_layers if isinstance(hidasi_layers, list)
                       else ast.literal_eval(hidasi_layers))
        if not self.layers or any(size <= 0 for size in self.layers):
            raise ValueError('hidasi_layers must contain positive integers')
        self.loss_name = hidasi_loss
        self.embedding_size = hidasi_embedding
        self.constrained_embedding = bool(hidasi_constrained_embedding)
        if self.constrained_embedding and self.embedding_size > 0:
            raise ValueError(
                'Set hidasi_embedding=0 with constrained embeddings')
        if not self.constrained_embedding and self.embedding_size <= 0:
            raise ValueError(
                'Set a positive hidasi_embedding for separate embeddings')
        self.dropout_p_embed = hidasi_dropout_embed
        self.dropout_p_hidden = hidasi_dropout_hidden
        self.momentum = hidasi_momentum
        self.bpreg = hidasi_bpreg
        self.elu_param = hidasi_elu_param
        self.adagrad_eps = hidasi_adagrad_eps
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        if self.constrained_embedding:
            input_size = self.layers[-1]
        else:
            input_size = self.embedding_size
            self.input_embeddings = nn.Embedding(
                self.item_num, input_size, sparse=True)
        self.output_embeddings = nn.Embedding(
            self.item_num, self.layers[-1], sparse=True)
        self.output_bias = nn.Embedding(self.item_num, 1, sparse=True)
        self.embedding_dropout = nn.Dropout(self.dropout_p_embed)

        cells = []
        dropouts = []
        for index, hidden_size in enumerate(self.layers):
            cell_input_size = (
                input_size if index == 0 else self.layers[index - 1])
            cells.append(nn.GRUCell(cell_input_size, hidden_size))
            dropouts.append(nn.Dropout(self.dropout_p_hidden))
        self.gru_cells = nn.ModuleList(cells)
        self.hidden_dropouts = nn.ModuleList(dropouts)
        self.reset_hidasi_parameters()

    @torch.no_grad()
    def reset_hidasi_parameters(self):
        if not self.constrained_embedding:
            init_parameter_matrix(self.input_embeddings.weight)
        init_parameter_matrix(self.output_embeddings.weight)
        nn.init.zeros_(self.output_bias.weight)
        for cell in self.gru_cells:
            init_parameter_matrix(cell.weight_ih, dim1_scale=3)
            init_parameter_matrix(cell.weight_hh, dim1_scale=3)
            nn.init.zeros_(cell.bias_ih)
            nn.init.zeros_(cell.bias_hh)

    def build_optimizer(self, learning_rate, l2_weight=0.0):
        return IndexedAdagradM(
            self.parameters(), lr=learning_rate,
            momentum=self.momentum, eps=self.adagrad_eps)

    def l2(self):
        # Hidasi's official PyTorch implementation removes L2 regularization.
        return next(self.parameters()).new_zeros(())

    def _input_embedding(self, item_ids):
        if self.constrained_embedding:
            return self.output_embeddings(item_ids)
        return self.input_embeddings(item_ids)

    def encode_history(self, history, lengths=None):
        if history.is_sparse:
            raise ValueError('GRU4RecHidasi requires --sparse_his 0')
        history = history.long().abs()
        if lengths is None:
            lengths = history.gt(0).long().sum(dim=1)
        lengths = lengths.long().to(history.device)
        batch_size, sequence_length = history.shape
        hidden = [self.output_embeddings.weight.new_zeros(
            (batch_size, size)) for size in self.layers]

        for step in range(sequence_length):
            active = step < lengths
            inputs = self.embedding_dropout(
                self._input_embedding(history[:, step]))
            for layer, cell in enumerate(self.gru_cells):
                updated = self.hidden_dropouts[layer](
                    cell(inputs, hidden[layer]))
                hidden[layer] = torch.where(
                    active.unsqueeze(1), updated, hidden[layer])
                inputs = hidden[layer]
        return hidden[-1]

    def score_candidates(self, encoded, candidate_ids):
        vectors = self.output_embeddings(candidate_ids)
        biases = self.output_bias(candidate_ids).squeeze(-1)
        return (encoded * vectors).sum(dim=1) + biases

    def score_matrix(self, encoded, candidate_ids):
        vectors = self.output_embeddings(candidate_ids)
        biases = self.output_bias(candidate_ids).view(1, -1)
        return torch.mm(encoded, vectors.t()) + biases

    def predict(self, feed_dict):
        item_ids = feed_dict[X][:, 1].long()
        encoded = self.encode_history(
            feed_dict[C_HISTORY], feed_dict.get(C_HISTORY_LENGTH))
        prediction = self.score_candidates(encoded, item_ids)
        return {
            'prediction': prediction,
            'check': [('prediction', prediction)],
            'embedding_l2': [],
        }

    def _bpr_max_loss(self, scores, batch_size):
        if self.elu_param > 0.0:
            scores = nn.functional.elu(scores, alpha=self.elu_param)
        negative_mask = torch.ones_like(scores)
        row_ids = torch.arange(batch_size, device=scores.device)
        negative_mask[row_ids, row_ids] = 0.0
        shifted = scores - scores.max(dim=1, keepdim=True)[0]
        weights = torch.exp(shifted) * negative_mask
        weights = weights / weights.sum(
            dim=1, keepdim=True).clamp(min=1e-24)
        targets = scores[row_ids, row_ids].unsqueeze(1)
        pairwise = torch.sigmoid(targets - scores)
        ranking = -torch.log(
            (pairwise * weights).sum(dim=1).clamp(min=1e-24))
        regularization = self.bpreg * (
            scores.pow(2) * weights).sum(dim=1)
        return (ranking + regularization).mean()

    def forward(self, feed_dict):
        if feed_dict['rank'] != 1:
            output = self.predict(feed_dict)
            output['loss'] = nn.functional.mse_loss(
                output['prediction'], feed_dict['Y'])
            return output

        total_size = feed_dict['Y'].shape[0]
        if total_size % 2 != 0:
            raise ValueError(
                'Ranking batches must contain positive/negative pairs')
        batch_size = total_size // 2
        lengths = feed_dict.get(C_HISTORY_LENGTH)
        if lengths is not None:
            lengths = lengths[:batch_size]
        encoded = self.encode_history(
            feed_dict[C_HISTORY][:batch_size], lengths)
        candidate_ids = feed_dict[X][:, 1].long()
        scores = self.score_matrix(encoded, candidate_ids)

        row_ids = torch.arange(batch_size, device=scores.device)
        positive_scores = scores[row_ids, row_ids]
        negative_scores = scores[row_ids, batch_size + row_ids]
        prediction = torch.cat([positive_scores, negative_scores])
        if self.loss_name == 'cross-entropy':
            loss = nn.functional.cross_entropy(scores, row_ids)
        else:
            loss = self._bpr_max_loss(scores, batch_size)
        return {
            'prediction': prediction,
            'loss': loss,
            'check': [('prediction', prediction)],
            'embedding_l2': [],
        }

