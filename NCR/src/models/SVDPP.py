# coding=utf-8

import torch
from models.RecModel import RecModel
from utils.global_p import X, C_HISTORY


class SVDPP(RecModel):
    data_processor = 'HisDataProcessor'

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.iid_embeddings_implicit = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'iid_embeddings_implicit',
                              'user_bias', 'item_bias']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[X][:, 0]
        i_ids = feed_dict[X][:, 1]
        history = feed_dict[C_HISTORY]

        if 'sparse' in str(history.type()):
            his_vector = history.mm(self.iid_embeddings_implicit.weight)
            his_length = torch.sparse.sum(history, dim=-1).to_dense()
        else:
            valid_his = history.abs().gt(0)
            history_abs = history.abs().long() * valid_his.long()
            his_vectors = self.iid_embeddings_implicit(history_abs)
            his_vector = (his_vectors * valid_his.unsqueeze(dim=-1).float()).sum(dim=1)
            his_length = valid_his.long().sum(dim=-1)

        valid_length = his_length.gt(0).float()
        safe_length = his_length.float() * valid_length + (1 - valid_length)
        his_vector = his_vector / safe_length.sqrt().view([-1, 1])

        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.extend([u_bias, i_bias, cf_u_vectors, cf_i_vectors, his_vector])

        prediction = ((cf_u_vectors + his_vector) * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias
        check_list.append(('prediction', prediction))

        return {'prediction': prediction,
                'check': check_list,
                'embedding_l2': embedding_l2}
