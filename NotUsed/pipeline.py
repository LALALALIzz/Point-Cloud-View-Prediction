import torch
import pandas
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Wingman import Helper
class Pipeline:

    # DatasetName: NAV, NJIT
    # mode: angular, position
    def __init__(self, datasetName, time_step, pred_step, num_feature, batch_size, random, mode, device):
        self.datasetName = datasetName
        self.time_step = time_step
        self.pred_step = pred_step
        self.num_feature = num_feature
        self.batch_size = batch_size
        self.random = random
        self.mode = mode
        self.device = device

    def load_dataset(self):
        if self.datasetName == "NAV":
            NavDataset = []
            for i in range(27):
                for j in range(4):
                    tmp_data = pandas.read_csv("../Data/P_%d/H%d_nav.csv" % (i+1, j+1)).to_numpy()
                    if self.mode == "position":
                        tmp_data = tmp_data[:, 2:5]
                    elif self.mode == "angular":
                        tmp_data = tmp_data[:, 5:8]
                        tmp_data = self.angular_encoder(tmp_data)
                    else:
                        print("wrong mode!")
                    NavDataset.append(tmp_data)
            return NavDataset
        elif self.datasetName == "NJIT":
            NjitDataset = []
            for i in range(18):
                tmp_data = pandas.read_csv("../NJIT/node%dmobility.csv" % (i+1)).to_numpy()
                if self.mode == "position":
                    tmp_data = tmp_data[:, 0:3]
                elif self.mode == "angular":
                    tmp_data = tmp_data[:, 3:6]
                    tmp_data = self.angular_encoder(tmp_data)
                else:
                    print("wrong mode!")
                NjitDataset.append(tmp_data)
            return NjitDataset
        else:
            print("Wrong DatasetName!")

    def Data_iter(self):
        raw_data = self.load_dataset()
        train_test_split_ratio = 0.6
        train_data = raw_data[:int(train_test_split_ratio*len(raw_data))]
        print(len(train_data))
        test_data = raw_data[int(train_test_split_ratio*len(raw_data)):]

        train_X = []
        train_Y = []
        for i in range(len(train_data)):
            pos = 0
            while(pos+self.time_step+self.pred_step < len(train_data[i])):
                train_X.append(train_data[i][pos:pos+self.time_step])
                train_Y.append(train_data[i][pos+self.time_step:pos+self.time_step+self.pred_step])
                pos += self.time_step
        train_set = CustomDataset(train_X, train_Y)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        test_X = []
        test_Y = []
        for i in range(len(test_data)):
            pos = 0
            while(pos+self.time_step+self.pred_step < len(test_data[i])):
                test_X.append(test_data[i][pos:pos+self.time_step])
                test_Y.append(test_data[i][pos+self.time_step:pos+self.time_step+self.pred_step])
                pos += self.time_step
        test_set = CustomDataset(test_X, test_Y)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train(self, model, train_loader, loss_func, optimizer):
        model.train()
        train_loss = 0
        count = 0
        for i, data in enumerate(train_loader):
            count = i+1
            X, Y = data
            X, Y = X.to(self.device), Y.to(self.device)
            pred = model(X)
            loss = loss_func(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.mean()
        train_loss = train_loss/count
        return train_loss

    def predict(self, model, test_loader):
        model.eval()
        test_loss = 0
        count = 0
        for i, data in enumerate(test_loader):
            count = i+1
            X, Y = data
            X, Y = X.to(self.device), Y.to(self.device)
            pred = model(X)
            '''
            print("*******************************")
            print(self.angular_decoder(Y))
            print("###############################")
            print(self.angular_decoder(pred))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            '''
            test_loss += Helper.degree_loss(pred, Y)
        test_loss = test_loss/count
        print(test_loss)
        return test_loss

    def automation(self, epoch, model, loss_func, optimizer, train_loader, test_loader, patience):
        train_loss = []
        test_loss = []
        trigger_cnt = 0
        last_loss = 300
        for e in tqdm(range(1, epoch + 1)):
            train_loss.append(self.train(model=model, train_loader=train_loader, loss_func=loss_func, optimizer=optimizer))
            current_loss = self.predict(model=model, test_loader=test_loader, loss_func=loss_func)
            test_loss.append(current_loss)
            if current_loss > last_loss:
                trigger_cnt += 1
                if trigger_cnt > patience:
                    torch.save(model.state_dict(), '../CHECKPOINTS/model2.pt')
                    Helper.plot_loss(e, train_loss, test_loss)
                    print('Early stopped!')
                    return model
            else:
                trigger_cnt = 0
            last_loss = current_loss
        print('Training finished!')
        Helper.plot_loss(epoch, train_loss, test_loss)
        torch.save(model.state_dict(), '../CHECKPOINTS/model2.pt')
        return model


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

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        Y = self.label[index]
        return X, Y

# if __name__ == '__main__':
#     datasetName = "NAV"
#     time_step = 30
#     pred_step = 30
#     batch_size = 32
#     num_feature = 6
#     num_hiddens = 20
#     num_layers = 1
#     learn_rate = 0.02
#     mode = "angular"
#     model = GRU_Dense(num_feature, num_hiddens, num_layers)
#     optimizer = optim.Adam(model.parameters(), lr=learn_rate)
#     criterion = nn.MSELoss()
#
#     pipline_test = Pipeline(datasetName=datasetName,
#                             time_step=time_step,
#                             pred_step=pred_step,
#                             num_feature=num_feature,
#                             batch_size=batch_size,
#                             random=1,
#                             mode=mode)
#     train_loader, test_loader = pipline_test.Data_iter()





