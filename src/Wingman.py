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
        for inputs, labels in test_loader:
            output = model(inputs)
            output_list.append(output[0, :, :].detach().numpy())
            label_list.append(labels[0, :, :].detach().numpy())
        return output_list, label_list

    @staticmethod
    def encdec_predict_test(model, test_loader, pred_step):
        model.eval()
        output_list = []
        label_list = []
        for encoder_inputs, (decoder_input, labels) in test_loader:
            enc_output, enc_state = model.encoder(encoder_inputs)
            dec_output, dec_state = model.decoder(encoder_inputs[:, -1, None], enc_state)
            dec_pred = dec_output
            for _ in range(pred_step - 1):
                dec_pred, dec_state = model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            output_list.append(dec_output[0, :, :].detach().numpy())
            label_list.append(labels[0, :, :].detach().numpy())
        return output_list, label_list







