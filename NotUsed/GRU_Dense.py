import torch
import torch.nn as nn

class GRU_Dense(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(GRU_Dense, self).__init__()
        self.model = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        self.tanh = nn.Tanh()
        self.dense1 = nn.Linear(hidden_dim, in_dim)

    def forward(self, inputs):
        outputs, _ = self.model(inputs)
        outputs = self.tanh(outputs)
        pred = self.dense1(outputs)
        pred = self.tanh(pred)
        return pred