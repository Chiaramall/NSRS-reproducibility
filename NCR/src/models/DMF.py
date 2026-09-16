# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils.global_p import X


class DMF(RecModel):
    """
    Deep Matrix Factorization baseline.

    DMF maps the user interaction vector and the item interaction vector through
    separate MLP towers, then scores a candidate with their dot product. The NCR
    runner supplies the pairwise BPR loss when --rank 1 is used.
    """

    @staticmethod
    def parse_model_args(parser, model_name='DMF'):
        parser.add_argument('--dmf_user_embedding_size', type=int, default=128,
                            help='First projection size for the user interaction vector.')
        parser.add_argument('--dmf_item_embedding_size', type=int, default=128,
                            help='First projection size for the item interaction vector.')
        parser.add_argument('--dmf_user_layers', type=str, default='[128]',
                            help='Hidden layers after the user projection.')
        parser.add_argument('--dmf_item_layers', type=str, default='[128]',
                            help='Hidden layers after the item projection.')
        parser.add_argument('--dmf_inter_matrix_type', type=str, default='01', choices=['01', 'rating'],
                            help='Use binary interactions or rating-valued interactions as DMF input.')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, train_df, label, dmf_user_embedding_size, dmf_item_embedding_size,
                 dmf_user_layers, dmf_item_layers, dmf_inter_matrix_type, *args, **kwargs):
        self.dmf_user_embedding_size = dmf_user_embedding_size
        self.dmf_item_embedding_size = dmf_item_embedding_size
        self.dmf_user_layers = dmf_user_layers if type(dmf_user_layers) == list else eval(dmf_user_layers)
        self.dmf_item_layers = dmf_item_layers if type(dmf_item_layers) == list else eval(dmf_item_layers)
        assert self.dmf_user_layers[-1] == self.dmf_item_layers[-1]
        self.dmf_inter_matrix_type = dmf_inter_matrix_type
        self.train_label = label
        self.user_num = kwargs['user_num']
        self.item_num = kwargs['item_num']
        self.user_history, self.item_history = self._build_histories(train_df, label)
        RecModel.__init__(self, *args, **kwargs)

    def _build_histories(self, train_df, label):
        user_history = [[] for _ in range(self.user_num)]
        item_history = [[] for _ in range(self.item_num)]
        for row in train_df.itertuples(index=False):
            uid = int(getattr(row, 'uid'))
            iid = int(getattr(row, 'iid'))
            rating = float(getattr(row, label)) if hasattr(row, label) else 1.0
            if rating <= 0:
                continue
            value = 1.0 if self.dmf_inter_matrix_type == '01' else rating
            user_history[uid].append((iid, value))
            item_history[iid].append((uid, value))
        return user_history, item_history

    def _init_weights(self):
        self.user_linear = torch.nn.Linear(self.item_num, self.dmf_user_embedding_size, bias=False)
        self.item_linear = torch.nn.Linear(self.user_num, self.dmf_item_embedding_size, bias=False)

        pre_size = self.dmf_user_embedding_size
        for i, layer_size in enumerate(self.dmf_user_layers):
            setattr(self, 'user_layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            pre_size = layer_size

        pre_size = self.dmf_item_embedding_size
        for i, layer_size in enumerate(self.dmf_item_layers):
            setattr(self, 'item_layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            pre_size = layer_size

    def _history_matrix(self, ids, histories, width, device):
        matrix = torch.zeros(ids.size(0), width, device=device)
        for row_idx, raw_id in enumerate(ids.detach().cpu().numpy().tolist()):
            entries = histories[int(raw_id)]
            if len(entries) == 0:
                continue
            cols = torch.tensor([c for c, _ in entries], dtype=torch.long, device=device)
            vals = torch.tensor([v for _, v in entries], dtype=matrix.dtype, device=device)
            matrix[row_idx, cols] = vals
        return matrix

    def _project_history(self, ids, histories, linear):
        """Apply a bias-free linear layer to sparse interaction histories."""
        flat_indices, flat_values, offsets = [], [], []
        for raw_id in ids.detach().cpu().numpy().tolist():
            offsets.append(len(flat_indices))
            entries = histories[int(raw_id)]
            flat_indices.extend(c for c, _ in entries)
            flat_values.extend(v for _, v in entries)

        if len(flat_indices) == 0:
            return linear.weight.new_zeros((ids.size(0), linear.out_features))

        device = linear.weight.device
        indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
        values = torch.tensor(flat_values, dtype=linear.weight.dtype, device=device)
        offsets = torch.tensor(offsets, dtype=torch.long, device=device)

        # For a multi-hot row x and bias-free Linear W, x @ W.T is exactly
        # the weighted sum of the selected columns of W.
        return F.embedding_bag(
            indices,
            linear.weight.t(),
            offsets,
            mode='sum',
            per_sample_weights=values
        )

    def _mlp(self, vector, prefix, layers, dropout):
        for i in range(0, len(layers)):
            vector = getattr(self, '%s_layer_%d' % (prefix, i))(vector)
            vector = F.relu(vector)
            vector = torch.nn.Dropout(p=dropout)(vector)
        return vector

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict[X][:, 0]
        i_ids = feed_dict[X][:, 1]
        user_vector = self._project_history(u_ids, self.user_history, self.user_linear)
        item_vector = self._project_history(i_ids, self.item_history, self.item_linear)
        user_vector = self._mlp(user_vector, 'user', self.dmf_user_layers, feed_dict['dropout'])
        item_vector = self._mlp(item_vector, 'item', self.dmf_item_layers, feed_dict['dropout'])

        prediction = F.cosine_similarity(user_vector, item_vector, dim=1, eps=1e-8).view([-1])
        check_list.append(('prediction', prediction))
        return {'prediction': prediction,
                'check': check_list,
                'embedding_l2': [user_vector, item_vector]}
