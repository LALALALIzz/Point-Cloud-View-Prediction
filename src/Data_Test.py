import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PCloudDataSet import Dataset
from RNN_Models import Encoder, Decoder, EncoderDecoder
from Loss_Function import LossFunc
import pandas
import numpy as np
import matplotlib.pyplot as plt
from TestKit import TestKit
from tqdm import tqdm




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
    pred_step = 30
    batch_size = 10
    num_feature = 6
    num_hiddens = 30
    num_layers = 1
    learn_rate = 0.001
    drop_out = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model construction
    encoder = Encoder(num_feature, num_hiddens, num_layers, drop_out)
    decoder = Decoder(num_feature, num_hiddens, num_layers, drop_out)
    model = EncoderDecoder(encoder, decoder)
    param_init(model)
    model.to(device)
    # Model optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.MSELoss()
    # TestKit initialization
    test_kit1 = TestKit(time_step=time_step,
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



    for epoch in tqdm(range(1, epoch_num + 1)):
        #print('Epoch %d:' % epoch)
        train_loss.append(test_kit1.train(train_mode='general', training_amt=20, train_index=[13, 17, 24]))
        test_loss.append(test_kit1.predict(test_mode='manual', test_amt=0, test_index=[3, 15]))

    train_loss = np.array(train_loss)

    test_loss = np.array(test_loss)
    print(np.min(test_loss))
    epoch = np.arange(1, epoch_num + 1)
    plt.figure()
    plt.plot(epoch, train_loss, 'b')
    plt.plot(epoch, test_loss, 'r')
    plt.show()
