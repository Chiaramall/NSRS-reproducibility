# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils.global_p import X


class NeuMF(RecModel):
    """
    Neural Matrix Factorization baseline for the NCR runner.

    The model follows Neural Collaborative Filtering: a GMF branch models the
    element-wise user/item interaction and an MLP branch models nonlinear
    interaction. For the NCR paper setup the MLP has three layers [32, 16, 8].
    The BaseRunner supplies pairwise BPR loss when --rank 1 is used.
    """

    @staticmethod
    def parse_model_args(parser, model_name='NeuMF'):
        parser.add_argument('--neumf_mlp_layers', type=str, default='[32,16,8]',
                            help='NeuMF MLP hidden layer sizes.')
        parser.add_argument('--neumf_mlp_vector_size', type=int, default=16,
                            help='User/item embedding size for the MLP branch. 16+16 matches the first 32-unit layer.')
        parser.add_argument('--neumf_mf_vector_size', type=int, default=64,
                            help='User/item embedding size for the GMF branch.')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, neumf_mlp_layers, neumf_mlp_vector_size, neumf_mf_vector_size, *args, **kwargs):
        self.neumf_mlp_layers = neumf_mlp_layers if type(neumf_mlp_layers) == list else eval(neumf_mlp_layers)
        self.neumf_mlp_vector_size = neumf_mlp_vector_size
        self.neumf_mf_vector_size = neumf_mf_vector_size
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.user_mf_embeddings = torch.nn.Embedding(self.user_num, self.neumf_mf_vector_size)
        self.item_mf_embeddings = torch.nn.Embedding(self.item_num, self.neumf_mf_vector_size)
        self.user_mlp_embeddings = torch.nn.Embedding(self.user_num, self.neumf_mlp_vector_size)
        self.item_mlp_embeddings = torch.nn.Embedding(self.item_num, self.neumf_mlp_vector_size)
        self.l2_embeddings = [
            'user_mf_embeddings', 'item_mf_embeddings',
            'user_mlp_embeddings', 'item_mlp_embeddings'
        ]

        pre_size = 2 * self.neumf_mlp_vector_size
        for i, layer_size in enumerate(self.neumf_mlp_layers):
            setattr(self, 'mlp_layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            pre_size = layer_size

        predict_size = self.neumf_mf_vector_size + self.neumf_mlp_layers[-1]
        self.predict_layer = torch.nn.Linear(predict_size, 1)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[X][:, 0]
        i_ids = feed_dict[X][:, 1]

        mf_u_vectors = self.user_mf_embeddings(u_ids)
        mf_i_vectors = self.item_mf_embeddings(i_ids)
        mlp_u_vectors = self.user_mlp_embeddings(u_ids)
        mlp_i_vectors = self.item_mlp_embeddings(i_ids)
        embedding_l2.extend([mf_u_vectors, mf_i_vectors, mlp_u_vectors, mlp_i_vectors])

        mf_output = mf_u_vectors * mf_i_vectors
        mlp_output = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for i in range(0, len(self.neumf_mlp_layers)):
            mlp_output = getattr(self, 'mlp_layer_%d' % i)(mlp_output)
            mlp_output = F.relu(mlp_output)
            mlp_output = torch.nn.Dropout(p=feed_dict['dropout'])(mlp_output)

        prediction_input = torch.cat([mf_output, mlp_output], dim=-1)
        prediction = self.predict_layer(prediction_input).view([-1])
        check_list.append(('prediction', prediction))

        return {'prediction': prediction,
                'check': check_list,
                'embedding_l2': embedding_l2}
