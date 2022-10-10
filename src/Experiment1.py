from DataProcess import DataProcess
from ModelTrainer import ModelTrainer
from ModelConstruct import Basic_GRU, Encoder, Decoder, EncoderDecoder, MLPDecoder
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
from ResultVisualization import ResultVisualization
from Wingman import Helper
import numpy as np
if __name__ == '__main__':
    # Experiment configuration
    EXPERIMENT_ID = 8
    MODEL_ID = 2
    para_id = 1
    MODEL_SAVE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                    '..',
                                                    'CHECKPOINTS',
                                                    'Experiment%dmodel%dpara%d.pt'))

    # Dataset related parameters
    dataset_name = 'njit'
    mode = 'position'
    architecture = 'enc_dec'
    observ_step = 250
    pred_step = 250
    batch_size = 128
    train_index = [1]
    test_index = [1]

    # Model related parameters
    input_dim = 3
    hidden_dim = 1024
    num_layers = 1
    batch_first = True
    dropout = 0
    encoder = Encoder(input_dim=input_dim,
                      hidden_dim=hidden_dim,
                      num_layers=num_layers,
                      batch_first=batch_first,
                      dropout=dropout)
    decoder = MLPDecoder(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         batch_first=batch_first,
                         dropout=dropout)
    model = EncoderDecoder(encoder, decoder)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Training parameters
    previous_loss = 0
    early_stop_cnter = 0
    EARLY_STOP_PATIENCE = 4
    epoch = 100

    # Result containers
    train_loss_list = []
    test_loss_list = []
    saved_encoder = Encoder(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)
    saved_decoder = MLPDecoder(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout=dropout)
    saved_model = EncoderDecoder(saved_encoder, saved_decoder)

    # Data process
    exp_process = DataProcess(dataset_name=dataset_name,
                              mode=mode,
                              architecture=architecture,
                              observ_step=observ_step,
                              pred_step=pred_step,
                              batch_size=batch_size)

    train_loader, test_loader = exp_process.dataloader_generation(train_index=train_index,
                                                                  test_index=test_index)

    # Model Training
    exp_trainer = ModelTrainer(model=model,
                               loss_func=loss_func,
                               optimizer=optimizer,
                               device=device,
                               pred_step=pred_step,
                               num_layers=num_layers,
                               hidden_dim=hidden_dim)

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
    result_visual = ResultVisualization(mode=mode,
                                        architecture=architecture,
                                        observ_step=observ_step,
                                        pred_step=pred_step)
    saved_model.load_state_dict(torch.load(MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id)))
    saved_model.eval()
    output_list, label_list = Helper.encdec_predict_test(saved_model, test_loader, pred_step, num_layers, hidden_dim)

    # Uncomment when split is not 0
    # output_list, label_list = np.tile(output_list, 3), np.tile(label_list, 3)
    result_visual.loss_plot(train_loss=train_loss_list, test_loss=test_loss_list)

    for label, output in zip(label_list, output_list):
        result_visual.data_plot(groundtruth=label, prediction=output, zoomInRange=0, zoomInBias=0, mean=exp_process.mean, std=exp_process.std)







