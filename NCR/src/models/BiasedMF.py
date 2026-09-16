# coding=utf-8

import torch
from models.RecModel import RecModel
from utils.global_p import X


class BiasedMF(RecModel):
    """
    BPR-MF baseline from the NCR paper.

    The runner supplies the BPR pairwise ranking loss when --rank 1 is used;
    this class provides the biased matrix-factorization scoring function.
    """

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_bias', 'item_bias']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[X][:, 0]
        i_ids = feed_dict[X][:, 1]

        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])
        embedding_l2.extend([u_bias, i_bias])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.extend([cf_u_vectors, cf_i_vectors])

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias
        check_list.append(('prediction', prediction))

        return {'prediction': prediction,
                'check': check_list,
                'embedding_l2': embedding_l2}
