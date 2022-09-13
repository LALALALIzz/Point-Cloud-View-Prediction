import torch
import numpy as np
import matplotlib.pyplot as plt
from RNN_Models import TestModel
import pandas as pd
from Wingman import Helper
class Evaluation:
    def __init__(self, model, dataset, downsample, pred_step):
        self.model = model
        self.dataset = dataset
        self.downsample = downsample
        self.pred_step = pred_step

    def fit(self):
        self.model.eval()
        pred = []
        flag = 1
        if self.downsample:
            label = self.dataset[30:870, :]
            x_axis = np.arange(30, 870)
            for index in range(0, 811, 30):
                input = Helper.angular_encoder(self.dataset[index: index + 30, :]).reshape(1, 30, 6)
                #print(input.shape)
                output = Helper.angular_decoder(self.model(torch.tensor(input)))
                pred.append(output[:self.pred_step - 30])
        else:
            label = self.dataset[250:, :]
            x_axis = np.arange(250, len(self.dataset))
            index = 0
            while(index + 250 + self.pred_step < 30000):
                input = Helper.angular_encoder(self.dataset[index: index+250, :]).reshape(1, 250, 6)
                #print(input.shape)
                output = Helper.angular_decoder(self.model(torch.tensor(input).float()))
                output = output.detach().numpy()
                output = output[0, :, :]
                pred.append(output)
                index += self.pred_step

        if self.downsample:
            pred = np.array(pred).reshape(840, 3)
            pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
            label_x, label_y, label_z = label[:, 0], label[:, 1], label[:, 2]
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(x_axis, pred_x, 'b')
            plt.plot(x_axis, label_x, 'r')
            plt.title('x-angle')
            plt.subplot(3, 1, 2)
            plt.plot(x_axis, pred_y, 'b')
            plt.plot(x_axis, label_y, 'r')
            plt.title('y-angle')
            plt.subplot(3, 1, 3)
            plt.plot(x_axis, pred_z, 'b')
            plt.plot(x_axis, label_z, 'r')
            plt.title('z-angle')
            plt.show()
            return
        pred = np.array(pred)
        pred = pred.reshape(pred.size//3, 3)
        label = label[:pred.size // 3, :]
        x_axis = x_axis[:pred.size // 3]
        pred[pred > 180] = pred[pred > 180] - 360
        pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
        label_x, label_y, label_z = label[:, 0], label[:, 1], label[:, 2]
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(x_axis, pred_x, 'b')
        plt.plot(x_axis, label_x, 'r')
        plt.title('x-angle')
        plt.subplot(3, 1, 2)
        plt.plot(x_axis, pred_y, 'b')
        plt.plot(x_axis, label_y, 'r')
        plt.title('y-angle')
        plt.subplot(3, 1, 3)
        plt.plot(x_axis, pred_z, 'b')
        plt.plot(x_axis, label_z, 'r')
        plt.title('z-angle')
        plt.show()
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(x_axis[10000:20000], pred_x[10000:20000], 'b')
        plt.plot(x_axis[10000:20000], label_x[10000:20000], 'r')
        plt.title('x-angle')
        plt.subplot(3, 1, 2)
        plt.plot(x_axis[10000:20000], pred_y[10000:20000], 'b')
        plt.plot(x_axis[10000:20000], label_y[10000:20000], 'r')
        plt.title('y-angle')
        plt.subplot(3, 1, 3)
        plt.plot(x_axis[10000:20000], pred_z[10000:20000], 'b')
        plt.plot(x_axis[10000:20000], label_z[10000:20000], 'r')
        plt.title('z-angle')
        plt.show()
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(x_axis[10000:11000], pred_x[10000:11000], 'b')
        plt.plot(x_axis[10000:11000], label_x[10000:11000], 'r')
        plt.title('x-angle')
        plt.subplot(3, 1, 2)
        plt.plot(x_axis[10000:11000], pred_y[10000:11000], 'b')
        plt.plot(x_axis[10000:11000], label_y[10000:11000], 'r')
        plt.title('y-angle')
        plt.subplot(3, 1, 3)
        plt.plot(x_axis[10000:11000], pred_z[10000:11000], 'b')
        plt.plot(x_axis[10000:11000], label_z[10000:11000], 'r')
        plt.title('z-angle')
        plt.show()

if __name__ == '__main__':
    '''
    time_step = 250
    pred_step = 250
    num_feature = 6
    num_layers = 1
    learn_rate = 0.0000002
    drop_out = 0
    d_model = 512
    num_head = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch = 1600
    # Model construction
    model = TestModel(num_feature, d_model, num_layers, num_head, time_step, pred_step)
    # param_init(model)
    checkpoint = torch.load('../CHECKPOINTs/model.pt')
    model.load_state_dict(checkpoint)
    '''
    model = torch.load('../CHECKPOINTS/transformer_NJIT_00001_500.pt')
    dataset = pd.read_csv('../NJIT/node17mobility.csv').to_numpy()
    dataset = dataset[:, 5:8]
    eval = Evaluation(model, dataset, 0)
    eval.fit()