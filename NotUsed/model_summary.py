import torch
import torch.nn as nn
from RNN_Models import TestModel
from torchinfo import summary

if __name__ == '__main__':
    num_features = 3
    num_head = 3
    num_layer = 1
    observation = 30
    pred = 150

    model = TestModel(num_features, num_layer, num_head, observation, pred)

    summary(model, input_size=(5, 30, 3))
