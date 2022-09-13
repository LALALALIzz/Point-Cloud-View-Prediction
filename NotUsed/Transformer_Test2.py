import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TransformerDateset import TransformerDataset
from RNN_Models import Encoder, Decoder, EncoderDecoder, TestModel
import pandas
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm


class TransformerTest2:

    def __init__(self, time_step, pred_step, model, loss_func, optim, num_feature, batch_size, device):
        self.time_step = time_step
        self.pred_step = pred_step
        self.model = model
        self.loss_func = loss_func
        self.num_feature = num_feature
        self.batch_size = batch_size
        self.optimizer = optim
        self.device = device

    def random_data_iter(self, dataset):
        X = []
        Y = []

        num_subseqs = (len(dataset) - self.pred_step - self.time_step) // self.time_step
        initial_indices = list(range(0, num_subseqs * self.time_step, self.time_step))



        def data(pos, time_step):
            return dataset[pos: pos + time_step]

        def label(pos, pred_step):
            return dataset[pos: pos + pred_step]

        for i in range(num_subseqs):
            X.append(data(i, self.time_step))
            Y.append(label(i + self.time_step, self.pred_step))
        X = torch.tensor(np.array(X, dtype='float64').reshape((len(X), self.time_step, self.num_feature)))
        Y = torch.tensor(np.array(Y, dtype='float64').reshape((len(Y), self.pred_step, self.num_feature)))
        my_set = TransformerDataset(X, Y)
        my_dataloader = DataLoader(my_set, batch_size=self.batch_size, shuffle=True)
        return my_dataloader

    def multi_loader_retriever(self, set_index):
        my_data = np.empty((0, 3))
        for index in set_index:
            temp_data = pandas.read_csv("../NJIT/node%dmobility.csv" % index).to_numpy()
            temp_data = temp_data[:, 3:6]
            if len(temp_data.shape) < 2:
                temp_data = temp_data[:, None]
            my_data = np.vstack((my_data, temp_data))
        my_data = self.angular_encoder(my_data)
        # temp_data[temp_data > 180] = temp_data[temp_data > 180] - 360
        my_loader = self.random_data_iter(my_data)
        return my_loader

    def train(self, train_loader):
        model = self.model
        loss_func = self.loss_func
        optimizer = self.optimizer
        model.train()
        train_loss = 0
        temp_loss = 0
        counter = 0
        for encoder_inputs, labels in train_loader:
            # print(len(data_loader.dataset))
            encoder_inputs, labels = encoder_inputs.to(self.device), labels.to(self.device)
            # encoder_inputs, decoder_inputs, labels = encoder_inputs.to(torch.double), decoder_inputs.to(torch.double), labels.to(torch.double)
            output = model(encoder_inputs)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            # print(loss)
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
            counter += 1
            # print(temp_loss)
            # print("dataloader---------------------------")
        train_loss = temp_loss / counter

        return train_loss

    def predict(self, test_loader):
        model = self.model
        loss_func = self.loss_func
        model.eval()
        test_loss = 0
        temp_loss = 0
        counter = 0
        for encoder_inputs, labels in test_loader:
            # print(encoder_inputs[:, -1, None].shape)
            encoder_inputs, labels = encoder_inputs.to(self.device), labels.to(self.device)
            output = model(encoder_inputs)
            output = self.angular_decoder(output.to('cpu')).detach().numpy()
            labels = self.angular_decoder(labels.to('cpu')).detach().numpy()
            loss = output - labels
            loss = np.abs(loss)
            loss[loss > 180] = 360 - loss[loss > 180]
            temp_loss += np.mean(loss)
            counter += 1
        test_loss += temp_loss / counter
        print(test_loss)
        return test_loss

    @staticmethod
    def angular_encoder(angular_inputs):
        angular_inputs = (angular_inputs * np.pi) / 180
        len, feature_num = angular_inputs.shape
        angle_encoded = np.empty((len, feature_num * 2))
        for index in range(0, feature_num * 2, 2):
            temp_angle = angular_inputs[:, index // 2].astype(float)
            temp_sin = np.sin(temp_angle)
            temp_cos = np.cos(temp_angle)
            angle_encoded[:, index] = temp_sin
            angle_encoded[:, index + 1] = temp_cos
        return angle_encoded

    @staticmethod
    def angular_decoder(sin_cos_inputs):
        batch_num, len, feature_num = sin_cos_inputs.shape
        angular_decoded = torch.empty((batch_num, len, feature_num // 2))
        for index in range(0, feature_num, 2):
            temp_sin = sin_cos_inputs[:, :, index]
            temp_cos = sin_cos_inputs[:, :, index + 1]
            temp_angle = torch.arctan2(temp_sin, temp_cos).detach().numpy()
            temp_angle[temp_angle < 0] = temp_angle[temp_angle < 0] + 2 * np.pi
            angular_decoded[:, :, index // 2] = torch.tensor(temp_angle)
        return (angular_decoded / np.pi) * 180

    def getDegLoss(self, pred, label):
        pred_deg = self.angular_decoder(pred.to('cpu')).detach().numpy()
        label_deg = self.angular_decoder(label.to('cpu')).detach().numpy()
        deg_loss = pred_deg - label_deg
        deg_loss = np.abs(deg_loss)
        deg_loss[deg_loss > 180] = 360 - deg_loss[deg_loss > 180]
        return deg_loss


if __name__ == '__main__':
    # Test constants
    random_seed = 0
    time_step = 250
    pred_step = 250
    batch_size = 10
    num_feature = 6
    num_layers = 1
    learn_rate = 0.000001
    drop_out = 0.2
    csv_index = 18
    partition_ratio = 0.3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model construction
    model = TestModel(num_feature, num_layers, num_feature, time_step, pred_step)
    # param_init(model)
    model.to(device)
    # Model optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.MSELoss()
    # TestKit2 initialization
    test_kit1 = TransformerTest2(time_step=time_step,
                                 pred_step=pred_step,
                                 model=model,
                                 loss_func=criterion,
                                 optim=optimizer,
                                 num_feature=num_feature,
                                 batch_size=batch_size,
                                 device=device)
    train_index = np.arange(1, 13)
    test_index = np.arange(13, 19)
    train_loader = test_kit1.multi_loader_retriever(train_index)
    test_loader = test_kit1.multi_loader_retriever(test_index)
    epoch_num = 500
    train_loss = []
    test_loss = []
    for epoch in tqdm(range(1, epoch_num + 1)):
        # print('Epoch %d:' % epoch)
        train_loss.append(test_kit1.train(train_loader))
        test_loss.append(test_kit1.predict(test_loader))
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    print(np.min(test_loss))
    epoch = np.arange(1, epoch_num + 1)
    plt.figure()
    plt.plot(epoch, train_loss, 'b')
    plt.plot(epoch, test_loss, 'r')
    plt.show()
