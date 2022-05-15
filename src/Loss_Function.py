import torch
import torch.nn as nn

class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()

    def forward(self, pred, label):
        radian_pred = (pred * torch.pi) / 180
        radian_label = (label * torch.pi) / 180
        pred_sin = torch.sin(radian_pred)
        pred_cos = torch.cos(radian_pred)
        label_sin = torch.sin(radian_label)
        label_cos = torch.cos(radian_label)


        return

