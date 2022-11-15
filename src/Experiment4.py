from DataProcess import DataProcess
from ModelTrainer import ModelTrainer
from ModelConstruct import Basic_GRU, Encoder, Decoder, MLPDecoder, EncoderDecoder
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
from ResultVisualization import ResultVisualization
from Wingman import Helper

if __name__ == '__main__':
    # Experiment configuration
    EXPERIMENT_ID = 14
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
    pred_step = 60
    batch_size = 128
    train_index = [1]
    test_index = [1]
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
    model = EncoderDecoder(encoder, decoder)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.apply(Helper.weight_init)

    # Training parameters
    previous_loss = float('inf')
    epoch = 300

    # Result containers
    train_loss_list = []
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
                                                                                valid_ratio=valid_ratio)

    # Model Training
    exp_trainer = ModelTrainer(model=model,
                               loss_func=loss_func,
                               optimizer=optimizer,
                               device=device,
                               pred_step=pred_step,
                               num_layers=num_layers,
                               hidden_dim=hidden_dim)

    for epoch in tqdm(range(epoch)):
        train_loss = exp_trainer.enc_dec_train4(train_loader=train_loader)
        valid_loss = exp_trainer.enc_dec_predict3(test_loader=valid_loader)
        if valid_loss < previous_loss:
            torch.save(model.state_dict(), MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id))
        previous_loss = valid_loss
        train_loss_list.append(train_loss)
        test_loss_list.append(valid_loss)

    # Result visualization
    result_visual = ResultVisualization(mode=mode,
                                        architecture=architecture,
                                        observ_step=observ_step,
                                        pred_step=pred_step)
    saved_model.load_state_dict(torch.load(MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id)))
    saved_model.eval()
    output_list, label_list = Helper.encdec_predict_test3(saved_model, test_loader, pred_step, num_layers, hidden_dim)
    result_visual.loss_plot(train_loss=train_loss_list, test_loss=test_loss_list)
    for output, label in zip(output_list, label_list):
        result_visual.data_plot(groundtruth=label, prediction=output, zoomInRange=0, zoomInBias=0)







