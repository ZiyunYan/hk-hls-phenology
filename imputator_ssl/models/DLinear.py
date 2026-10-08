import torch
import torch.nn as nn



class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in


        self.Linear = nn.Linear(self.seq_len, self.seq_len)



    def forward(self, x, x_mark=None, missing_mask=None, ano_mask=None):
        # x: [Batch, Input length, Channel]


        x = self.Linear(x.permute(0,2,1))
        return x.permute(0, 2, 1), x.permute(0, 2, 1), x.permute(0, 2, 1), x.permute(0, 2, 1)
