# coding=utf-8

import math

import torch
import torch.nn.functional as F

from models.RecModel import RecModel
from utils.global_p import *


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)

    def forward(self, inputs, dropout):
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = F.dropout(outputs, p=dropout, training=self.training)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.dropout(outputs, p=dropout, training=self.training)
        return outputs.transpose(-1, -2)


class SASRec(RecModel):
    data_processor = 'ProLogicRecDP'
    uses_custom_initialization = True

    @staticmethod
    def parse_model_args(parser, model_name='SASRec'):
        parser.add_argument('--sasrec_num_blocks', type=int, default=2)
        parser.add_argument('--sasrec_num_heads', type=int, default=2)
        parser.add_argument('--sasrec_norm_first', type=int, default=0,
                            choices=[0, 1])
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, sasrec_num_blocks, sasrec_num_heads,
                 sasrec_norm_first, *args, **kwargs):
        self.num_blocks = sasrec_num_blocks
        self.num_heads = sasrec_num_heads
        self.norm_first = bool(sasrec_norm_first)
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        if self.ui_vector_size % self.num_heads != 0:
            raise ValueError('u/i vector size must be divisible by heads')

        self.iid_embeddings = torch.nn.Embedding(
            self.item_num, self.ui_vector_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(
            512, self.ui_vector_size, padding_idx=0)
        self.embedding_dropout = torch.nn.Dropout()

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        for _ in range(self.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(self.ui_vector_size, eps=1e-8))
            self.attention_layers.append(torch.nn.MultiheadAttention(
                self.ui_vector_size, self.num_heads, dropout=0.0))
            self.forward_layernorms.append(
                torch.nn.LayerNorm(self.ui_vector_size, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(self.ui_vector_size))
        self.last_layernorm = torch.nn.LayerNorm(self.ui_vector_size, eps=1e-8)
        self.l2_embeddings = ['iid_embeddings', 'position_embeddings']
        self._reset_parameters()

    def _reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.01)
        with torch.no_grad():
            self.iid_embeddings.weight[0].fill_(0.0)
            self.position_embeddings.weight[0].fill_(0.0)

    def _history_vectors(self, history, dropout):
        valid_his = history.abs().gt(0)
        history_abs = history.abs() * valid_his.long()
        seqs = self.iid_embeddings(history_abs)
        seqs = seqs * math.sqrt(self.ui_vector_size)

        positions = torch.arange(
            1, history.size(1) + 1, device=history.device).view(1, -1)
        positions = positions.expand_as(history_abs) * valid_his.long()
        seqs = seqs + self.position_embeddings(positions)
        seqs = torch.nn.Dropout(p=dropout)(seqs)
        seqs = seqs * valid_his.unsqueeze(-1).float()

        seq_len = seqs.size(1)
        attention_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=seqs.device),
            diagonal=1)

        for i in range(self.num_blocks):
            seqs_t = seqs.transpose(0, 1)
            if self.norm_first:
                query = self.attention_layernorms[i](seqs_t)
                attn_out, _ = self.attention_layers[i](
                    query, query, query, attn_mask=attention_mask)
                seqs = (seqs_t + attn_out).transpose(0, 1)
                seqs = seqs + self.forward_layers[i](
                    self.forward_layernorms[i](seqs), dropout)
            else:
                attn_out, _ = self.attention_layers[i](
                    seqs_t, seqs_t, seqs_t, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](
                    (seqs_t + attn_out).transpose(0, 1))
                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs, dropout))
            seqs = seqs * valid_his.unsqueeze(-1).float()
        return self.last_layernorm(seqs), valid_his

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        i_ids = feed_dict[X][:, 1]
        history = feed_dict[C_HISTORY]

        log_feats, valid_his = self._history_vectors(
            history, feed_dict['dropout'])
        his_length = valid_his.long().sum(dim=-1)
        batch_idx = torch.arange(history.size(0), device=history.device)
        last_idx = (his_length - 1).clamp(min=0)
        final_feat = log_feats[batch_idx, last_idx, :]

        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.extend([final_feat, cf_i_vectors])
        prediction = (final_feat * cf_i_vectors).sum(dim=-1).view([-1])
        prediction = torch.clamp(prediction, min=-20.0, max=20.0)
        check_list.append(('prediction', prediction))

        return {'prediction': prediction, 'check': check_list,
                'embedding_l2': embedding_l2}
