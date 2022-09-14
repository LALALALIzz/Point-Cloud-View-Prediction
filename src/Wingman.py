import numpy as np
import matplotlib.pyplot as plt
import torch

class Helper:

    @staticmethod
    def plot_loss(epoch, train_loss, test_loss):
        x_axis = np.arange(1, epoch + 1)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, train_loss, 'b')
        plt.title('Train Loss')
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, test_loss, 'r')
        plt.title('Test Loss')
        plt.show()

    @staticmethod
    def angular_encoder(angular_inputs):
        angular_inputs = (angular_inputs * np.pi) / 180
        len, feature_num = angular_inputs.shape
        angle_encoded = np.empty((len, feature_num * 2))
        for index in range(0, feature_num * 2, 2):
            temp_angle = angular_inputs[:, index // 2].astype('float64')
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

    @staticmethod
    def degree_loss(pred, label):
        pred_deg = Helper.angular_decoder(pred.to('cpu')).detach().numpy()
        label_deg = Helper.angular_decoder(label.to('cpu')).detach().numpy()
        deg_loss = pred_deg - label_deg
        deg_loss = np.abs(deg_loss)
        deg_loss[deg_loss > 180] = 360 - deg_loss[deg_loss > 180]
        return np.mean(deg_loss)

    @staticmethod
    def basic_predict_test(model, test_loader):
        model.eval()
        output_list = []
        label_list = []
        ymin = [1000, 1000, 1000]
        ymax = [-1000, -1000, -1000]

        """
        inputs shape is (batch, in_len, 3)
        output shape is (batch, out_len, 3)
        in_flatten shape is (batch*in_len, 3)
        out_flatten shape is (batch*out_len, 3)
        data_concate shape is (batch*in_len+batch*out_len, 3)
        """
        for inputs, labels in test_loader:
            output = model(inputs)
            in_flatten = torch.flatten(inputs, start_dim=0, end_dim=1)
            # out_flatten = torch.flatten(output,start_dim=0, end_dim=1)
            # data_concate = torch.cat((in_flatten,out_flatten), 0)
            tmp_min = torch.min(in_flatten, dim=0).values.detach().numpy()
            tmp_max = torch.max(in_flatten, dim=0).values.detach().numpy()
            for i in range(len(ymax)):
                if tmp_min[i] < ymin[i]:
                    ymin[i] = tmp_min[i]
                if tmp_max[i] > ymax[i]:
                    ymax[i] = tmp_max[i]
            output_list.append(output[0, :, :].detach().numpy())
            label_list.append(np.concatenate((inputs[0, :, :].detach().numpy(), labels[0, :, :].detach().numpy()), axis=0))
        return output_list, label_list, ymin, ymax

    @staticmethod
    def encdec_predict_test(model, test_loader, pred_step):
        model.eval()
        output_list = []
        label_list = []
        ymin = [1000, 1000, 1000]
        ymax = [-1000, -1000, -1000]

        """
        inputs shape is (batch, in_len, 3)
        output shape is (batch, out_len, 3)
        in_flatten shape is (batch*in_len, 3)
        out_flatten shape is (batch*out_len, 3)
        data_concate shape is (batch*in_len+batch*out_len, 3)
        """
        for encoder_inputs, (decoder_input, labels) in test_loader:
            enc_output, enc_state = model.encoder(encoder_inputs)
            dec_output, dec_state = model.decoder(encoder_inputs[:, -1, None], enc_state)
            dec_pred = dec_output
            for _ in range(pred_step - 1):
                dec_pred, dec_state = model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            in_flatten = torch.flatten(encoder_inputs, start_dim=0, end_dim=1)
            # out_flatten = torch.flatten(dec_output,start_dim=0, end_dim=1)
            # data_concate = torch.cat((in_flatten, out_flatten), 0)
            tmp_min = torch.min(in_flatten, dim=0).values.detach().numpy()
            tmp_max = torch.max(in_flatten, dim=0).values.detach().numpy()
            for i in range(len(ymax)):
                if tmp_min[i] < ymin[i]:
                    ymin[i] = tmp_min[i]
                if tmp_max[i] > ymax[i]:
                    ymax[i] = tmp_max[i]
            output_list.append(dec_output[0, :, :].detach().numpy())
            label_list.append(np.concatenate((encoder_inputs[0, :, :].detach().numpy(), labels[0, :, :].detach().numpy()), axis=0))
        return output_list, label_list, ymin, ymax








