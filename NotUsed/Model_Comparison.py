import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PCloudDataSet import Dataset
from RNN_Models import Encoder, Decoder, EncoderDecoder, BaselineLinear, BaselineGRU
from Loss_Function import LossFunc
import pandas
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


class BaselineModel:

    def __init__(self, time_step, pred_step, model, loss_func, optim, num_feature, batch_size, random, device):
        self.time_step = time_step
        self.pred_step = pred_step
        self.model = model
        self.loss_func = loss_func
        self.num_feature = num_feature
        self.batch_size = batch_size
        self.optimizer = optim
        self.random = random
        self.device = device

    def random_data_iter(self, dataset):
        X = []
        Y = []
        num_subseqs = (len(dataset) - self.pred_step - self.time_step) // self.time_step
        initial_indices = list(range(0, num_subseqs * self.time_step, self.time_step))

        if self.random == 1:
            random.shuffle(initial_indices)

        def data(pos, time_step):
            return dataset[pos: pos + time_step]

        def label(pos, pred_step):
            return dataset[pos: pos + pred_step]

        for i in range(num_subseqs):
            X.append(data(i, self.time_step))
            Y.append(label(i + self.time_step, self.pred_step))
        X = torch.tensor(np.array(X, dtype='float32').reshape((len(X), self.time_step, self.num_feature)))
        Y = torch.tensor(np.array(Y, dtype='float32').reshape((len(Y), self.pred_step, self.num_feature)))
        my_set = Dataset(X, Y)
        my_dataloader = DataLoader(my_set, batch_size=self.batch_size, shuffle=False)
        return my_dataloader

    def multi_loader_retriever(self, mode, series_amt, manual_index):
        series_index = []
        if mode == 'manual':
            series_index = manual_index
        else:
            series_index = np.arange(1, series_amt + 1)
        for index in series_index:
            temp_data = pandas.read_csv("../Data/P_%d/H1_nav.csv" % index).to_numpy()
            temp_data = temp_data[:, 5:8]
            if len(temp_data.shape) < 2:
                temp_data = temp_data[:, None]
            temp_data = self.angular_encoder(temp_data)
            # temp_data[temp_data > 180] = temp_data[temp_data > 180] - 360
            temp_data_loader = self.random_data_iter(temp_data)
            if len(temp_data_loader.dataset) > 0:
                yield temp_data_loader

    def train(self, train_mode, training_amt, train_index):
        model = self.model
        loss_func = self.loss_func
        optimizer = self.optimizer
        model.train()
        train_loss = 0
        for data_loader in self.multi_loader_retriever(mode=train_mode, series_amt=training_amt,
                                                       manual_index=train_index):
            temp_loss = 0
            for encoder_inputs, (decoder_inputs, labels) in data_loader:
                encoder_inputs, decoder_inputs, labels = encoder_inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
                # pass in encoder_inputs for baseline model
                dec_output = model(encoder_inputs)
                loss = loss_func(dec_output, labels[:, :encoder_inputs.shape[1], :])
                optimizer.zero_grad()
                # print(loss)
                loss.backward()
                optimizer.step()
                temp_loss += loss.item() / encoder_inputs.shape[0]
                # print(temp_loss)
            train_loss += temp_loss / len(data_loader.dataset)
            # print("dataloader---------------------------")
        if train_mode == 'manual':
            train_loss = train_loss / len(train_index)
        else:
            train_loss = train_loss / training_amt
        train_loss = train_loss / self.time_step
        # print('Model trained with train loss: %.3f' % train_loss )

        return train_loss

    def predict(self, test_mode, test_amt, test_index):
        model = self.model
        loss_func = self.loss_func
        model.eval()
        test_loss = 0
        for data_loader in self.multi_loader_retriever(mode=test_mode, series_amt=test_amt, manual_index=test_index):
            temp_loss = 0
            for encoder_inputs, (decoder_input, labels) in data_loader:
                encoder_inputs, decoder_input, labels = encoder_inputs.to(self.device), decoder_input.to(self.device), labels.to(self.device)
                dec_output = model(encoder_inputs)
                # print(dec_output.shape)
                # print(dec_output.shape)
                # print('---')
                # print(labels.shape)
                # print('*************')
                # loss = loss_func(dec_output * 1000, labels * 1000)
                # temp_loss = loss.item() / encoder_inputs.shape[0]

                # print(dec_output)
                # print('-----------')
                # print(labels)
                # print('batch-end')
                for _ in range((self.pred_step - 30)//30):
                    first_output = dec_output[:, -self.time_step:, :]
                    temp_output = model(first_output)
                    dec_output = torch.cat((dec_output, temp_output), dim=1)
                    #print(dec_output.shape)
                dec_output = self.angular_decoder(dec_output.to('cpu')).detach().numpy()
                labels = self.angular_decoder(labels.to('cpu')).detach().numpy()

                loss = dec_output - labels
                loss = np.abs(loss)
                loss[loss > 180] = 360 - loss[loss > 180]

                temp_loss += np.sum(loss) / encoder_inputs.shape[0]

            test_loss += temp_loss / len(data_loader.dataset)
        if test_mode == 'manual':
            test_loss = test_loss / len(test_index)
        else:
            test_loss = test_loss / test_amt
        test_loss = test_loss / self.pred_step
        # print('Model tested with test loss: %.3f' % test_loss)
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

    def param_init(model):
        for name, param in model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

if __name__ == '__main__':
    torch.manual_seed(0)
    # Test constants
    time_step = 30
    pred_step = 150
    batch_size = 15
    num_feature = 6
    num_hiddens = 30
    num_layers = 1
    learn_rate = 0.01
    drop_out = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model construction
    #model = BaselineLinear(num_feature, num_feature)

    model = BaselineGRU(num_feature, num_hiddens, num_layers, drop_out)
    #param_init(model)
    model.to(device)
    # Model optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.MSELoss()
    # TestKit2 initialization
    test_kit1 = BaselineModel(time_step=time_step,
                              pred_step=pred_step,
                              model=model,
                              loss_func=criterion,
                              optim=optimizer,
                              num_feature=num_feature,
                              batch_size=batch_size,
                              random=0,
                              device=device)

    epoch_num = 100
    train_loss = []
    test_loss = []

    #pred_num = 30
    for epoch in tqdm(range(1, epoch_num + 1)):
        # print('Epoch %d:' % epoch)
        train_loss.append(test_kit1.train(train_mode='general', training_amt=25, train_index=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]))
        test_loss.append(test_kit1.predict(test_mode='manual', test_amt=0, test_index=[26]))

    train_loss = np.array(train_loss)

    test_loss = np.array(test_loss)
    print(np.min(test_loss))
    epoch = np.arange(1, epoch_num + 1)
    plt.figure()
    plt.plot(epoch, train_loss, 'b')
    plt.plot(epoch, test_loss, 'r')
    plt.show()
