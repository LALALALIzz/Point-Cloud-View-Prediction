import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import numpy as np
from Wingman import Helper
from CustomDataset import BasicDataset, EncDecDataset, EncDecDataset2
import os
import scipy.signal as ss

class DataProcess:

    def __init__(self, dataset_name, mode, architecture, observ_step, pred_step, batch_size, separation=None):
        self.dataset_name = dataset_name
        self.mode = mode
        self.architecture = architecture
        self.observ_step = observ_step
        self.pred_step = pred_step
        self.batch_size = batch_size
        self.separation = separation
        self.FEATURE_NUM = 3
        self.mean = None
        self.std = None

    def Decimator(self, dataset):
        decimated = []
        for i in range(dataset.shape[1]):
            decimated.append(ss.decimate(dataset[:, i], 5))
        decimated = np.stack(decimated, axis=1)
        return decimated

    def Normalization(self, dataset):
        mean = []
        std = []
        for i in range(dataset.shape[1]):
            mean.append(np.mean(dataset[:, i]))
            std.append(np.std(dataset[:, i]))
            dataset[:, i] = (dataset[:, i] - np.mean(dataset[:, i])) / np.std(dataset[:, i])
        self.mean = mean
        self.std = std

    def Window_normalization(self, X, Y):
        x_wn = np.zeros((X.shape[0], X.shape[1]))
        y_wn = np.zeros((Y.shape[0], Y.shape[1]))
        for i in range(X.shape[1]):
            head = X[0, i]
            std = np.std(X[:, i])
            x_wn[:, i] = (X[:, i] - head) / std
            y_wn[:, i] = (Y[:, i] - head) / std
        return x_wn, y_wn

        self.mean = mean
        self.std = std

    def dataset_generation(self, csv_path):
        # Initialize input, label
        # Read csv file
        X = []
        Y = []
        dataset = pd.read_csv(csv_path).to_numpy()
        # self.Normalization(dataset)
        # Retrieve data from csv file based on dataset name and mode
        # dataset name in {umd, njit}
        # mode in {position, angle}
        # if mode is 'angle', pass through angular encoder
        if self.dataset_name == 'umd':
            if self.mode == 'position':
                dataset = dataset[:, 2:5]
                if self.separation is not None:
                    dataset = dataset[:, self.separation]
            elif self.mode == 'angle':
                dataset = dataset[:, 5:8]
                if self.separation is not None:
                    dataset = dataset[:, self.separation]
                dataset = Helper.angular_encoder(dataset)
            else:
                raise Exception("Mode should be in {position, angle}")
        elif self.dataset_name == 'njit':
            if self.mode == 'position':
                dataset = dataset[:5000, :3]
                if self.separation is not None:
                    dataset = dataset[:, self.separation]
            elif self.mode == 'angle':
                dataset = dataset[:, 3:6]
                if self.separation is not None:
                    dataset = dataset[:, self.separation]
                dataset = Helper.angular_encoder(dataset)
            else:
                raise Exception("Mode should be in {position, angle}")
        else:
            raise Exception("Dataset name should be in {'umd', 'njit'}")
        # Add a dummy dimension if dataset is 1-d array
        if len(dataset.shape) < 2:
            dataset = dataset[:, None]

        # dataset = self.Decimator(dataset)
        self.Normalization(dataset)

        # Get sequence start indices
        num_samples = (len(dataset) - self.pred_step - self.observ_step+1)
        initial_indices = np.arange(0, num_samples)
        # Helper function to separate input and label
        def data(pos, observ_step):
            return dataset[pos: pos + observ_step]
        def label(pos, pred_step):
            return dataset[pos: pos + pred_step]
        # Fill X with input sequence
        # Fill Y with label sequence
        for i in initial_indices:  # range(len(dataset) - self.pred_step - self.time_step):
            x_wn, y_wn = self.Window_normalization(data(i, self.observ_step), label(i + self.observ_step, self.pred_step))
            X.append(x_wn)
            Y.append(y_wn)
        # Generate different dataset according to achitecture and separation
        # architecture in {basic, enc_dec}
        # separation in {None, 0, 1, 2}
        if self.separation is not None:
            feature_num = 1
        else:
            feature_num = self.FEATURE_NUM
        X = torch.tensor(np.array(X, dtype='float32').reshape((len(X), self.observ_step, feature_num)))
        Y = torch.tensor(np.array(Y, dtype='float32').reshape((len(Y), self.pred_step, feature_num)))
        if self.architecture == 'basic':
            my_set = BasicDataset(X, Y)
        elif self.architecture == 'enc_dec':
            my_set = EncDecDataset2(X, Y)
        else:
            raise Exception("Architecture should be in {'basic', 'enc_dec'}")
        # print(len(my_set))
        return my_set

    def dataloader_generation(self, train_index, test_index):
        trainset_list = []
        testset_list = []
        if self.dataset_name == 'umd':
            csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'P_%d', 'H1_nav.csv'))
        elif self.dataset_name == 'njit':
            csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'NJIT', 'node%dmobility.csv'))
        else:
            raise Exception("Dataset name should be in {umd, njit}")
        # Concatenate all sub-datasets
        for index in train_index:
            trainset_path = csv_path % index
            trainset_list.append(self.dataset_generation(trainset_path))
        trainset = ConcatDataset(trainset_list)
        for index in test_index:
            testset_path = csv_path % index
            testset_list.append(self.dataset_generation(testset_path))
        testset = ConcatDataset(testset_list)
        train_loader = DataLoader(trainset,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(testset,
                                 batch_size=self.batch_size,
                                 shuffle=False)
        return train_loader, test_loader



