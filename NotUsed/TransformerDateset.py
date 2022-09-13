import torch

class TransformerDataset(torch.utils.data.Dataset):
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

