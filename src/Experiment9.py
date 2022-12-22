from DataProcess import DataProcess
from ModelTrainer import ModelTrainer
from ModelConstruct import *
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
from ResultVisualization import ResultVisualization
from Wingman import Helper

if __name__ == '__main__':
    # Experiment configuration
    EXPERIMENT_ID = 17
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
    observ_step = 50
    pred_step = 50
    batch_size = 512
    train_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    valid_index = [13]
    test_index = [15, 18]
    valid_ratio = 0.9
    # Model related parameters
    input_dim = 3
    hidden_dim = 512
    num_layers = 2
    batch_first = True
    dropout = 0
    encoder = Encoder(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       batch_first=batch_first,
                       dropout=dropout)
    decoder = Decoder(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       batch_first=batch_first,
                       dropout=dropout)
    loss_func = nn.MSELoss()
    enc_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    decoder.to(device)

    # Training parameters
    previous_loss = float('inf')
    epochs = 300
    # Result containers
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    saved_encoder = Encoder(input_dim=input_dim,
                             hidden_dim=hidden_dim,
                             num_layers=num_layers,
                             batch_first=batch_first,
                             dropout=dropout)
    saved_decoder = Decoder(input_dim=input_dim,
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

    train_loader, valid_loader, test_loader = exp_process.dataloader_generation(train_index=train_index,
                                                                                test_index=test_index,
                                                                                validate_index=valid_index)

    # Model Training
    model_trainer = ModelTrainer(model=encoder,
                               loss_func=loss_func,
                               optimizer=enc_optimizer,
                               device=device,
                               pred_step=pred_step,
                               num_layers=num_layers,
                               hidden_dim=hidden_dim)

    for epoch in tqdm(range(epochs)):
        train_loss = model_trainer.decoder_train5(train_loader=train_loader)
        valid_loss = model_trainer.enc_dec_predict4(test_loader=valid_loader)
        if valid_loss < previous_loss:
            torch.save(model.state_dict(), MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id))
            print(valid_loss)
            previous_loss = valid_loss
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    print("Decoder is trained.")

    # Result visualization
    result_visual = ResultVisualization(mode=mode,
                                        architecture=architecture,
                                        observ_step=observ_step,
                                        pred_step=pred_step)
    print("Model is predicting...")
    saved_model.load_state_dict(torch.load(MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id)))
    saved_model.eval()
    output_list, enc_output_list, label_list = Helper.encdec_predict_test5(saved_model, test_loader, pred_step, num_layers, hidden_dim)
    result_visual.loss_plot(train_loss=enc_train_loss_list, test_loss=enc_valid_loss_list)
    result_visual.loss_plot(train_loss=train_loss_list, test_loss=valid_loss_list)
    for (output, enc_output), label in zip(zip(output_list, enc_output_list), label_list):
        result_visual.data_plot2(groundtruth=label, enc_output=enc_output, prediction=output, zoomInRange=0, zoomInBias=0)








