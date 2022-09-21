import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.inputs[index]
        Y = self.labels[index]
        return X, Y


class EncDecDataset(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.inputs[index]

        x_T = X[-1]
        x_T = x_T[None, :]

        y_label = self.labels[index]
        y_inputs = torch.cat((x_T, y_label[:-1, :]))
        return X, (y_inputs, y_label)
