import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from TransformerDateset import TransformerDataset
from RNN_Models import TestModel
from RNN_Models import Transformer, AlteredGRU
# from Loss_Function import LossFunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from scipy import signal
from Wingman import Helper
from Model_Eval import Evaluation

class TransformerTest:

    def __init__(self, time_step, pred_step, model, loss_func, optim, mode, num_feature, batch_size, device):
        self.time_step = time_step
        self.pred_step = pred_step
        self.model = model
        self.loss_func = loss_func
        self.mode = mode
        self.batch_size = batch_size
        self.optimizer = optim
        self.num_feature = num_feature
        self.device = device

    def dataset_generation(self, csv_path):
        # Initialize input, label
        # Read csv file
        X = []
        Y = []
        dataset = pd.read_csv(csv_path).to_numpy()
        '''
        dataset = []
        
        for i in range(0, len(init_dataset), 5):
            dataset.append(init_dataset[i, :])
        dataset = np.array(dataset)
        '''
        # Retrieve data from csv file based on mode
        # mode in ['angular', 'coord']

        if self.mode == 'angular':
            dataset = Helper.angular_encoder(dataset[:, 3:6])
        elif self.mode == 'coord':
            dataset = dataset[:, :3]
        # Add dummy dimension to prevent error
        if len(dataset.shape) < 2:
            dataset = dataset[:, None]
        # Get sequence start indices
        num_subseqs = (len(dataset) - self.pred_step - self.time_step) // self.time_step
        initial_indices = np.arange(0, num_subseqs * self.time_step, self.time_step)
        # Helper function to separate input and label
        def data(pos, time_step):
            return dataset[pos: pos + time_step]
        def label(pos, pred_step):
            return dataset[pos: pos + pred_step]
        # Fill X with input sequence
        # Fill Y with label sequence
        for i in initial_indices: #range(len(dataset) - self.pred_step - self.time_step):
            X.append(data(i, self.time_step))
            Y.append(label(i + self.time_step, self.pred_step))
        X = torch.tensor(np.array(X, dtype='float32').reshape((len(X), self.time_step, self.num_feature)))
        Y = torch.tensor(np.array(Y, dtype='float32').reshape((len(Y), self.pred_step, self.num_feature)))
        my_set = TransformerDataset(X, Y)
        #print(len(my_set))
        return my_set

    def dataloader_generation(self, train_index, test_index):
        trainset_list = []
        testset_list = []
        # Concatenate all sub-datasets
        for index in train_index:
            trainset_path = "../NJIT/node%dmobility.csv" % index
            trainset_list.append(self.dataset_generation(trainset_path))
        trainset = ConcatDataset(trainset_list)
        for index in test_index:
            testset_path = "../NJIT/node%dmobility.csv" % index
            testset_list.append(self.dataset_generation(testset_path))
        testset = ConcatDataset(testset_list)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        return train_loader, test_loader

    def train(self, train_loader):
        model = self.model
        loss_func = self.loss_func
        optimizer = self.optimizer
        model.train()
        train_loss = 0
        temp_loss = 0
        counter = 0
        for encoder_inputs, labels in train_loader:
            encoder_inputs, labels = encoder_inputs.to(self.device), labels.to(self.device)
            output = model(encoder_inputs)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
            counter += 1
        train_loss = temp_loss / counter
        #print(train_loss)
        return train_loss

    def predict(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for encoder_inputs, labels in test_loader:
            # print(encoder_inputs[:, -1, None].shape)
            encoder_inputs, labels = encoder_inputs.to(self.device), labels.to(self.device)
            output = self.model(encoder_inputs)
            if self.mode == 'angular':
                test_loss += Helper.degree_loss(output, labels)
            else:
                test_loss += self.loss_func(output, labels).item()
            counter+=1
        test_loss = test_loss / counter
        print(test_loss)
        return test_loss

    def automation(self, epoch, train_index, test_index, patience, model_name, model_index, testset):
        train_loss = []
        test_loss = []
        trigger_cnt = 0
        last_loss = 300
        train_loader, test_loader = self.dataloader_generation(train_index, test_index)
        for e in tqdm(range(1, epoch + 1)):
            train_loss.append(self.train(train_loader))
            current_loss = self.predict(test_loader)
            test_loss.append(current_loss)
            if current_loss > last_loss:
                trigger_cnt += 1
                if trigger_cnt > patience:
                    torch.save(self.model.state_dict(), '../CHECKPOINTS/%s%d.pt' % (model_name, model_index))
                    Helper.plot_loss(e, train_loss, test_loss)
                    print('Early stopped!')
                    self.model.to('cpu')
                    eval = Evaluation(self.model, testset, 0, pred_step)
                    eval.fit()
                    return self.model
            else:
                trigger_cnt = 0
            last_loss = current_loss
        print('Training finished!')
        Helper.plot_loss(epoch, train_loss, test_loss)
        torch.save(self.model.state_dict(), '../CHECKPOINTS/%s%d.pt' % (model_name, model_index))
        self.model.to('cpu')
        eval = Evaluation(self.model, testset, 0, pred_step)
        print(len(testset))
        eval.fit()
        return min(test_loss), self.model



if __name__ == '__main__':

        torch.cuda.empty_cache()
        # Test constants
        time_step = 250
        batch_size = 5
        num_feature = 6
        num_layers = 1
        learn_rate = 0.00001
        d_model = 512
        num_head2 = 6
        num_head1 = 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        epoch = 500
        num_hidden = 250
        pred_list = [250, 500, 750, 1000, 1250]
        train_index = np.arange(1, 13)
        test_index = np.arange(13, 19)
        criterion = nn.MSELoss()
        GRU_Loss = []
        testset1 = pd.read_csv('../NJIT/node17mobility.csv').to_numpy()
        testset1 = testset1[:, 3:6]
        '''
        for pred_step in pred_list:
            torch.cuda.empty_cache()
            model1 = AlteredGRU(in_dim=num_feature,
                                hidden_dim=num_hidden,
                                num_layers=num_layers,
                                observ_step=time_step,
                                pred_step=pred_step)
            # param_init(model)

            model1.to(device)
            model1.float()
            optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
            test_kit1 = TransformerTest(time_step=time_step,
                                        pred_step=pred_step,
                                        model=model1,
                                        loss_func=criterion,
                                        optim=optimizer,
                                        mode='angular',
                                        batch_size=batch_size,
                                        num_feature=num_feature,
                                        device=device)
            min_loss, _ = test_kit1.automation(epoch=epoch,
                                               train_index=train_index,
                                               test_index=test_index,
                                               patience=5,
                                               model_name='GRU',
                                               model_index=pred_step,
                                               testset=testset1)
            GRU_Loss.append(min_loss)
        
        Transformer1_Loss = []
        for pred_step in [1000, 1250]:
            torch.cuda.empty_cache()
            model2 = Transformer(num_features=num_feature,
                                 num_layers=num_layers,
                                 num_head=num_head2,
                                 observ_step=time_step,
                                 pred_step=pred_step)
            # param_init(model)
            model2.to(device)
            optimizer = optim.Adam(model2.parameters(), lr=learn_rate)
            test_kit1 = TransformerTest(time_step=time_step,
                                        pred_step=pred_step,
                                        model=model2,
                                        loss_func=criterion,
                                        optim=optimizer,
                                        mode='angular',
                                        batch_size=batch_size,
                                        num_feature=num_feature,
                                        device=device)
            min_loss, _ = test_kit1.automation(epoch=epoch,
                                               train_index=train_index,
                                               test_index=test_index,
                                               patience=5,
                                               model_name='Transformer1',
                                               model_index=pred_step,
                                               testset=testset1)
            Transformer1_Loss.append(min_loss)

        '''
        Transformer2_Loss = []
        for pred_step in pred_list:
            torch.cuda.empty_cache()
            model3 = TestModel(num_features=num_feature,
                               d_model=d_model,
                               num_layers=num_layers,
                               num_head=num_head1,
                               observ_step=time_step,
                               pred_step=pred_step)
            # param_init(model)
            model3.to(device)
            optimizer = optim.Adam(model3.parameters(), lr=learn_rate)
            test_kit1 = TransformerTest(time_step=time_step,
                                        pred_step=pred_step,
                                        model=model3,
                                        loss_func=criterion,
                                        optim=optimizer,
                                        mode='angular',
                                        batch_size=batch_size,
                                        num_feature=num_feature,
                                        device=device)
            min_loss, _ = test_kit1.automation(epoch=epoch,
                                               train_index=train_index,
                                               test_index=test_index,
                                               patience=5,
                                               model_name='Transformer2',
                                               model_index=pred_step,
                                               testset=testset1)
            Transformer2_Loss.append(min_loss)
