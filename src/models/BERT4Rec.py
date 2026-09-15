# coding=utf-8

import math

import torch
import torch.nn.functional as F

from models.RecModel import RecModel
from utils.global_p import *


class GELU(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BERTFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BERTFeedForward, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermediate_size)
        self.activation = GELU()
        self.dense_out = torch.nn.Linear(intermediate_size, hidden_size)

    def forward(self, x, dropout):
        x = self.activation(self.dense_in(x))
        x = F.dropout(x, p=dropout, training=self.training)
        return self.dense_out(x)


class BERT4RecBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout):
        super(BERT4RecBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0)
        self.attention_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.forward_layer = BERTFeedForward(
            hidden_size, intermediate_size)
        self.forward_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

    def forward(self, seqs, key_padding_mask, dropout):
        seqs_t = seqs.transpose(0, 1)
        attn_out, _ = self.attention(
            seqs_t, seqs_t, seqs_t,
            key_padding_mask=key_padding_mask)
        seqs = self.attention_layernorm(
            seqs + F.dropout(attn_out.transpose(0, 1), p=dropout, training=self.training))
        ff_out = self.forward_layer(seqs, dropout)
        seqs = self.forward_layernorm(
            seqs + F.dropout(ff_out, p=dropout, training=self.training))
        return seqs


class BERT4Rec(RecModel):
    data_processor = 'ProLogicRecDP'
    uses_custom_initialization = True

    @staticmethod
    def parse_model_args(parser, model_name='BERT4Rec'):
        parser.add_argument('--bert_num_blocks', type=int, default=2)
        parser.add_argument('--bert_num_heads', type=int, default=2)
        parser.add_argument('--bert_intermediate_size', type=int, default=256)
        parser.add_argument('--bert_pooling', type=str, default='last',
                            choices=['last', 'mean'])
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, bert_num_blocks, bert_num_heads,
                 bert_intermediate_size, bert_pooling, *args, **kwargs):
        self.num_blocks = bert_num_blocks
        self.num_heads = bert_num_heads
        self.intermediate_size = bert_intermediate_size
        self.pooling = bert_pooling
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        if self.ui_vector_size % self.num_heads != 0:
            raise ValueError('u/i vector size must be divisible by heads')

        self.iid_embeddings = torch.nn.Embedding(
            self.item_num, self.ui_vector_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(
            512, self.ui_vector_size, padding_idx=0)
        self.embedding_layernorm = torch.nn.LayerNorm(
            self.ui_vector_size, eps=1e-8)
        self.blocks = torch.nn.ModuleList([
            BERT4RecBlock(
                self.ui_vector_size, self.num_heads,
                self.intermediate_size, 0.0)
            for _ in range(self.num_blocks)
        ])
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
        seqs = self.embedding_layernorm(seqs)
        seqs = torch.nn.Dropout(p=dropout)(seqs)
        seqs = seqs * valid_his.unsqueeze(-1).float()

        key_padding_mask = ~valid_his
        for block in self.blocks:
            seqs = block(seqs, key_padding_mask, dropout)
            seqs = seqs * valid_his.unsqueeze(-1).float()
        return self.last_layernorm(seqs), valid_his

    def _pool_sequence(self, seqs, valid_his):
        his_length = valid_his.long().sum(dim=-1)
        if self.pooling == 'mean':
            denom = his_length.clamp(min=1).view(-1, 1).float()
            return (seqs * valid_his.unsqueeze(-1).float()).sum(dim=1) / denom
        batch_idx = torch.arange(seqs.size(0), device=seqs.device)
        last_idx = (his_length - 1).clamp(min=0)
        return seqs[batch_idx, last_idx, :]

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        i_ids = feed_dict[X][:, 1]
        history = feed_dict[C_HISTORY]

        seqs, valid_his = self._history_vectors(history, feed_dict['dropout'])
        user_vector = self._pool_sequence(seqs, valid_his)
        user_vector = F.normalize(user_vector, dim=-1)

        cf_i_vectors = self.iid_embeddings(i_ids)
        item_vector = F.normalize(cf_i_vectors, dim=-1)
        embedding_l2.extend([user_vector, cf_i_vectors])

        prediction = (user_vector * item_vector).sum(dim=-1).view([-1])
        prediction = torch.clamp(prediction, min=-20.0, max=20.0)
        check_list.append(('prediction', prediction))

        return {'prediction': prediction, 'check': check_list,
                'embedding_l2': embedding_l2}
