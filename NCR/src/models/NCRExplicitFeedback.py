# coding=utf-8
"""
NCR with Explicit Feedback (Section 4.2 of the paper)
Uses positive/negative feedback signals explicitly in the logical reasoning.

Formula: e_v1_u ∧ e_v2_u ∧ · · · ∧ ¬e_vr_u → e_vx_u
Equivalent to: (¬e_v1_u ∨ ¬e_v2_u ∨ · · · ∨ ¬¬e_vr_u ) ∨ e_vx_u
"""

import torch
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils import utils
import numpy as np
from utils import global_p


class NCRExplicitFeedback(BaseModel):
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='NCRExplicitFeedback'):
        parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
        parser.add_argument('--r_weight', type=float, default=10,
                            help='Weight of logic regularizer loss')
        parser.add_argument('--ppl_weight', type=float, default=0,
                            help='Weight of uv interaction prediction loss')
        parser.add_argument('--pos_weight', type=float, default=0,
                            help='Weight of positive purchase loss')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 r_weight, ppl_weight, pos_weight, random_seed, model_path):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        self.r_weight = r_weight
        self.ppl_weight = ppl_weight
        self.pos_weight = pos_weight
        self.sim_scale = 10
        BaseModel.__init__(self, label_min=label_min, label_max=label_max,
                           feature_num=feature_num, random_seed=random_seed,
                           model_path=model_path)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 0.1, size=self.ui_vector_size).astype(np.float32)), requires_grad=False)
        self.not_layer_1 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.not_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.and_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.and_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.or_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.or_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_1 = torch.nn.Linear(2 * self.ui_vector_size, self.ui_vector_size)
        self.purchase_layer_2 = torch.nn.Linear(self.ui_vector_size, self.ui_vector_size)

    def logic_not(self, vector):
        """Logical NOT operation: ¬x"""
        vector = F.relu(self.not_layer_1(vector))
        vector = self.not_layer_2(vector)
        return vector

    def logic_and(self, vector1, vector2):
        """Logical AND operation: x ∧ y"""
        assert(len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.and_layer_1(vector))
        vector = self.and_layer_2(vector)
        return vector

    def logic_or(self, vector1, vector2):
        """Logical OR operation: x ∨ y"""
        assert (len(vector1.size()) == len(vector2.size()))
        vector = torch.cat((vector1, vector2), dim=(len(vector1.size()) - 1))
        vector = F.relu(self.or_layer_1(vector))
        vector = self.or_layer_2(vector)
        return vector

    def purchase_gate(self, uv_vector):
        """Gate that combines user and item embeddings"""
        uv_vector = F.relu(self.purchase_layer_1(uv_vector))
        uv_vector = self.purchase_layer_2(uv_vector)
        return uv_vector

    def mse(self, vector1, vector2):
        return ((vector1 - vector2) ** 2).mean()

    def predict(self, feed_dict):
        """
        Prediction with explicit feedback reasoning.
        Uses history_pos_tag to distinguish e_v_u (positive) from ¬e_v_u (negative).
        
        For e_v_u = 1: represents e_v_u (user liked item v)
        For e_v_u = 0: represents ¬e_v_u (user disliked item v)
        """
        check_list = []
        u_ids = feed_dict['X'][:, 0]
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[global_p.C_HISTORY]
        batch_size, his_length = list(history.size())

        # Explicit feedback: 1=positive, 0=negative
        history_pos_tag = feed_dict[global_p.C_HISTORY_POS_TAG].unsqueeze(2).float()

        # Embeddings
        user_vectors = self.uid_embeddings(u_ids)
        item_vectors = self.iid_embeddings(i_ids)

        # Prepare candidate item: concat uid + iid
        item_vectors = torch.cat((user_vectors, item_vectors), dim=1)
        item_vectors = self.purchase_gate(item_vectors)

        # Expand user vector for broadcasting
        uh_vectors = user_vectors.view(user_vectors.size(0), 1, user_vectors.size(1))
        uh_vectors = uh_vectors.expand(history_pos_tag.size(0), history_pos_tag.size(1), uh_vectors.size(2))

        # History item embeddings
        his_vectors = self.iid_embeddings(history)
        his_vectors = torch.cat((uh_vectors, his_vectors), dim=2)
        his_vectors = self.purchase_gate(his_vectors)
        
        # Compute representations for both e_v_u and ¬e_v_u
        not_his_vectors = self.logic_not(his_vectors)

        # EXPLICIT FEEDBACK LOGIC:
        # history_pos_tag == 1 -> use his_vectors (e_v_u: user liked v)
        # history_pos_tag == 0 -> use not_his_vectors (¬e_v_u: user disliked v)
        his_vectors_explicit = history_pos_tag * his_vectors + (1 - history_pos_tag) * not_his_vectors

        # constraint tracks the logical statements for regularization
        constraint = [his_vectors_explicit, not_his_vectors]

        # Build antecedent with negated history using DNF: ¬e_v1_u ∨ ¬e_v2_u ∨ ... ∨ e_vr_u
        # Shuffle history indices for variability
        shuffled_history_idx = [i for i in range(his_length)]
        np.random.shuffle(shuffled_history_idx)
        
        # Start with ¬(first history item's explicit feedback)
        tmp_vector = self.logic_not(his_vectors_explicit[:, shuffled_history_idx[0]])
        
        # OR together ¬ of remaining history items
        for i in shuffled_history_idx[1:]:
            tmp_vector = self.logic_or(tmp_vector, self.logic_not(his_vectors_explicit[:, i]))
            constraint.append(tmp_vector.view(batch_size, -1, self.ui_vector_size))
        
        left_vector = tmp_vector  # Antecedent: ¬e_v1_u ∨ ¬e_v2_u ∨ ... ∨ ¬¬e_vr_u

        # Consequent: e_vx_u (the candidate item)
        right_vector = item_vectors
        constraint.append(right_vector.view(batch_size, -1, self.ui_vector_size))
        
        # Final implication: antecedent ∨ consequent
        sent_vector = self.logic_or(left_vector, right_vector)
        constraint.append(sent_vector.view(batch_size, -1, self.ui_vector_size))

        if feed_dict['rank'] == 1:
            prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * 10
        else:
            prediction = F.cosine_similarity(sent_vector, self.true.view([1, -1])) * \
                         (self.label_max - self.label_min) / 2 + (self.label_max + self.label_min) / 2

        constraint = torch.cat(constraint, dim=1)
        out_dict = {'prediction': prediction,
                    'check': check_list,
                    'constraint': constraint,
                    'interim': left_vector}
        return out_dict

    def forward(self, feed_dict):
        """
        Forward pass: compute predictions and loss with logical regularization.
        """
        out_dict = self.predict(feed_dict)
        check_list = out_dict['check']
        false = self.logic_not(self.true).view(1, -1)
        constraint = out_dict['constraint']

        dim = len(constraint.size())-1

        # ===== LOGICAL REGULARIZATION =====
        # Constraints to ensure logical operations are well-defined
        
        # NOT: ¬¬x = x
        r_not_not_true = (1 - F.cosine_similarity(self.logic_not(self.logic_not(self.true)), self.true, dim=0)).sum()
        r_not_not_self = \
            (1 - F.cosine_similarity(self.logic_not(self.logic_not(constraint)), constraint, dim=dim)).mean()

        # NOT: ¬x ≠ x (orthogonality)
        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraint), constraint, dim=dim)).mean()

        # NOT NOT NOT: ¬¬¬x = ¬x
        r_not_not_not = \
            (1 + F.cosine_similarity(self.logic_not(self.logic_not(constraint)), self.logic_not(constraint), dim=dim)).mean()

        # AND: x ∧ True = x
        r_and_true = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.true.expand_as(constraint)), constraint, dim=dim)).mean()
        
        # AND: x ∧ False = False
        r_and_false = (1 - F.cosine_similarity(
            self.logic_and(constraint, false.expand_as(constraint)), false.expand_as(constraint), dim=dim)).mean()
        
        # AND: x ∧ x = x
        r_and_self = (1 - F.cosine_similarity(self.logic_and(constraint, constraint), constraint, dim=dim)).mean()

        # AND: x ∧ ¬x = False (contradiction)
        r_and_not_self = (1 - F.cosine_similarity(
            self.logic_and(constraint, self.logic_not(constraint)), false.expand_as(constraint), dim=dim)).mean()
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_and(self.logic_not(constraint), constraint), false.expand_as(constraint), dim=dim)).mean()

        # OR: x ∨ True = True
        r_or_true = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.true.expand_as(constraint)), self.true.expand_as(constraint), dim=dim)).mean()
        
        # OR: x ∨ False = x
        r_or_false = (1 - F.cosine_similarity(
            self.logic_or(constraint, false.expand_as(constraint)), constraint, dim=dim)).mean()
        
        # OR: x ∨ x = x
        r_or_self = (1 - F.cosine_similarity(self.logic_or(constraint, constraint), constraint, dim=dim)).mean()

        # OR: x ∨ ¬x = True (tautology)
        r_or_not_self = (1 - F.cosine_similarity(
            self.logic_or(constraint, self.logic_not(constraint)), self.true.expand_as(constraint), dim=dim)).mean()
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_or(self.logic_not(constraint), constraint), self.true.expand_as(constraint), dim=dim)).mean()

        # True/False orthogonality
        true_false = 1 + F.cosine_similarity(self.true, false.view(-1), dim=0)

        # Aggregate regularization loss
        r_loss = r_not_not_true + r_not_not_self + r_not_self + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse + \
                 r_or_true + r_or_false + r_or_self + true_false + r_or_not_self + r_or_not_self_inverse + r_not_not_not
        r_loss = r_loss * self.r_weight

        # ===== RECOMMENDATION LOSS =====
        if feed_dict['rank'] == 1:
            # BPR loss for ranking task
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            pos, neg = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
            loss = -(pos - neg).sigmoid().log().sum()
        else:
            # MSE loss for rating prediction
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])

        # Total loss
        loss = loss + r_loss
        out_dict['loss'] = loss
        out_dict['check'] = check_list
        return out_dict
