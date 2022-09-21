import matplotlib.pyplot as plt
import numpy as np
import torch
from CustomDataset import BasicDataset, EncDecDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from ModelConstruct import EncoderDecoder, Encoder, Decoder
from ModelTrainer import ModelTrainer
from ResultVisualization import ResultVisualization
from Wingman import Helper
from tqdm import tqdm
import os

def generate_testset(num_points, observ_step, pred_step, feature_num, architecture):
    gaussian_noise = np.random.normal(0, 1, num_points)
    x_values1 = np.linspace(-1500, 1500, num_points)
    x_values2 = x_values1 + 100
    x_values3 = x_values1 - 100
    sin_values1 = np.sin(x_values1) + gaussian_noise
    cos_values1 = (np.cos(x_values2) + gaussian_noise) / 100
    sin_values2 = np.sin(x_values3) + gaussian_noise
    dataset = np.transpose(np.vstack((sin_values1, cos_values1, sin_values2)))
    # print(dataset.shape)
    # dataset = sin_values + gaussian_noise
    #dataset = sin_values
    X, Y =[], []
    # Get sequence start indices
    num_samples = (len(dataset) - pred_step - observ_step) // observ_step
    initial_indices = np.arange(0, num_samples * observ_step, observ_step)

    # Helper function to separate input and label
    def data(pos, observ_step):
        return dataset[pos: pos + observ_step]

    def label(pos, pred_step):
        return dataset[pos: pos + pred_step]

    # Fill X with input sequence
    # Fill Y with label sequence
    for i in initial_indices:  # range(len(dataset) - self.pred_step - self.time_step):
        X.append(data(i, observ_step))
        Y.append(label(i + observ_step, pred_step))
    # Generate different dataset according to achitecture and separation
    # architecture in {basic, enc_dec}
    # separation in {None, 0, 1, 2}
    X = torch.tensor(np.array(X, dtype='float32').reshape((len(X), observ_step, feature_num)))
    Y = torch.tensor(np.array(Y, dtype='float32').reshape((len(Y), pred_step, feature_num)))
    if architecture == 'basic':
        my_set = BasicDataset(X, Y)
    elif architecture == 'enc_dec':
        my_set = EncDecDataset(X, Y)

    return my_set

if __name__ == '__main__':
    num_points = 30000
    observ_step = 250
    pred_step = 250
    feature_num = 3
    architecture = 'enc_dec'
    batch_size = 32

    hidden_dim = 700
    num_layers = 1
    batch_first = True
    dropout = 0

    EXPERIMENT_ID = 3

    MODEL_ID = 1

    para_id = 1

    MODEL_SAVE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                    '..',
                                                    'CHECKPOINTS',
                                                    'Experiment%dmodel%dpara%d.pt'))


    dataset = generate_testset(num_points=num_points,
                               observ_step=observ_step,
                               pred_step=pred_step,
                               feature_num=feature_num,
                               architecture=architecture)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True)

    encoder = Encoder(input_dim=feature_num,
                      hidden_dim=hidden_dim,
                      num_layers=num_layers,
                      batch_first=batch_first,
                      dropout=dropout)
    decoder = Decoder(input_dim=feature_num,
                      hidden_dim=hidden_dim,
                      num_layers=num_layers,
                      batch_first=batch_first,
                      dropout=dropout)
    model = EncoderDecoder(encoder, decoder)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)

    # Training parameters
    previous_loss = 0
    early_stop_cnter = 0
    EARLY_STOP_PATIENCE = 4
    epoch = 80

    # Result containers
    train_loss_list = []
    test_loss_list = []
    saved_encoder = Encoder(input_dim=feature_num,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)
    saved_decoder = Decoder(input_dim=feature_num,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)
    saved_model = EncoderDecoder(saved_encoder, saved_decoder)

    train_loader, test_loader = data_loader, data_loader

    # Model Training
    exp_trainer = ModelTrainer(model=model,
                               loss_func=loss_func,
                               optimizer=optimizer,
                               device=device,
                               pred_step=pred_step)

    for epoch in tqdm(range(epoch)):
        train_loss = exp_trainer.enc_dec_train(train_loader=train_loader)
        test_loss = exp_trainer.enc_dec_predict(test_loader=test_loader)
        if test_loss > previous_loss:
            early_stop_cnter += 1
            if early_stop_cnter >= EARLY_STOP_PATIENCE:
                break
        else:
            early_stop_cnter = 0
        previous_loss = test_loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    torch.save(model.state_dict(), MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id))

    # Result visualization
    result_visual = ResultVisualization(mode='position',
                                        architecture=architecture,
                                        observ_step=observ_step,
                                        pred_step=pred_step)
    saved_model.load_state_dict(torch.load(MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id)))
    saved_model.eval()
    output_list, label_list, ymin, ymax = Helper.encdec_predict_test(saved_model, test_loader, pred_step)
    result_visual.loss_plot(train_loss=train_loss_list, test_loss=test_loss_list)
    for output, label in zip(output_list, label_list):
        result_visual.data_plot(groundtruth=label, prediction=output, zoomInRange=0, zoomInBias=0, ymin=ymin, ymax=ymax)

    '''
    # Result visualization
    saved_model.load_state_dict(torch.load(MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id)))
    saved_model.eval()
    output_list, label_list = Helper.encdec_predict_test(saved_model, test_loader, pred_step)
    for output, label in zip(output_list, label_list):
        #print(label.shape)
        x = np.arange(observ_step, observ_step + pred_step)
        y = np.arange(observ_step + pred_step)
        plt.plot(x, output, 'o')
        plt.plot(y, label, 'b')
        plt.show()
    '''


