import torch
from torch.utils.data import Dataset
import numpy as np

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

class EncDecDataset2(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.inputs[index]

        x_inputs = X[:-1]
        x_T = X[-1]
        x_T = x_T[None, :]

        y_label = self.labels[index]
        y_inputs = torch.cat((x_T, y_label[:-1, :]))
        return x_inputs, (y_inputs, y_label)


class EncDecDataset3(Dataset):
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

        y_label = torch.cat((X[-10:], self.labels[index]), dim=0)
        y_inputs = torch.cat((x_T, y_label[:-1, :]), dim=0)
        return X, (y_inputs, y_label)

# Week 8
# Experiment 14 dataset
class EncDecDataset4(Dataset):
    def __init__(self, inputs, labels, start_pos):
        self.labels = labels
        self.inputs = inputs
        self.start_pos = start_pos

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        Y = self.labels[index]

        # Use 40 data points as encoder inputs
        # Use 40-50 as start token
        # Calculate loss on 50-100
        encoder_inputs = X[:50-self.start_pos]
        start_token = X[50-self.start_pos:50]
        labels = torch.concat((start_token, Y), dim=0)

        return encoder_inputs, (start_token, labels)

# Week 8
# Experiment 15 dataset
class EncDecDataset5(Dataset):
    def __init__(self, inputs, labels, start_pos):
        self.labels = labels
        self.inputs = inputs
        self.start_pos = start_pos

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        Y = self.labels[index]

        # Use 40 data points as encoder inputs
        # Use 40-50 as start token
        # Calculate loss on 50-100
        encoder_inputs = X
        encoder_labels = torch.concat((X[1:], Y[None, 0]), dim=0)
        labels = Y

        return encoder_inputs, (encoder_labels, labels)


# Week 8
# Experiment 16 dataset
class EncDecDataset6(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        Y = self.labels[index]

        # Use 40 data points as encoder inputs
        # Use 40-50 as start token
        # Calculate loss on 50-100
        encoder_inputs = X
        encoder_labels = torch.concat((X[1:], Y[None, 0]), dim=0)
        decoder_inputs = torch.concat((X[None, -1], Y[:-1]), dim=0)
        labels = Y

        return encoder_inputs, (encoder_labels, decoder_inputs, labels)