# coding=utf-8

import torch

from models.NCRExplicitFeedback import NCRExplicitFeedback


class NCREwoLR(NCRExplicitFeedback):
    """NCR-E without logical regularization (lambda_r = 0)."""

    def __init__(self, *args, **kwargs):
        kwargs['r_weight'] = 0.0
        super(NCREwoLR, self).__init__(*args, **kwargs)

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)

        if feed_dict['rank'] == 1:
            batch_size = int(feed_dict['Y'].shape[0] / 2)
            pos = out_dict['prediction'][:batch_size]
            neg = out_dict['prediction'][batch_size:]
            loss = -(pos - neg).sigmoid().log().sum()
        else:
            loss = torch.nn.MSELoss()(out_dict['prediction'], feed_dict['Y'])

        out_dict['loss'] = loss
        return out_dict
