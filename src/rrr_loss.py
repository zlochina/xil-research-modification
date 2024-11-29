import torch.nn as nn

class RRRLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RRRLoss, self).__init__(*args, **kwargs)

    def forward(self, predictions, targets, binary_mask):
        loss = 0
        return 0