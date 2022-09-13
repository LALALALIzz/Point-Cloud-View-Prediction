from DataProcess import DataProcess
from ModelTrainer import ModelTrainer
from ModelConstruct import Basic_GRU
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
from ResultVisualization import ResultVisualization
from Wingman import Helper

if __name__ == '__main__':
    # Experiment configuration
    EXPERIMENT_ID = 1
    MODEL_ID = 1
    para_id = 0
    MODEL_SAVE_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                    '..',
                                                    'CHECKPOINTS',
                                                    'Experiment%dmodel%dpara%id.pt'))

    # Dataset related parameters
    dataset_name = 'njit'
    mode = 'position'
    architecture = 'basic'
    observ_step = 30
    pred_step = 30
    batch_size = 10
    train_index = [1]
    test_index = [1]

    # Model related parameters
    input_dim = 3
    hidden_dim = 128
    num_layers = 3
    batch_first = True
    dropout = 0.2
    model = Basic_GRU(input_dim=input_dim,
                      hidden_dim=hidden_dim,
                      num_layers=num_layers,
                      batch_first=batch_first,
                      dropout=dropout,
                      pred_step=pred_step)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Training parameters
    previous_loss = 0
    early_stop_cnter = 0
    EARLY_STOP_PATIENCE = 3
    epoch = 1

    # Result containers
    train_loss_list = []
    test_loss_list = []
    saved_model = Basic_GRU(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout,
                            pred_step=pred_step)

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
                               pred_step=pred_step)

    for epoch in tqdm(range(epoch)):
        train_loss = exp_trainer.basic_train(train_loader=train_loader)
        test_loss = exp_trainer.basic_predict(test_loader=test_loader, test_mode=0)
        if test_loss > previous_loss:
            early_stop_cnter += 1
            if early_stop_cnter >= EARLY_STOP_PATIENCE:
                break
        else:
            early_stop_cnter = 0
        previous_loss = test_loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    else:
        torch.save(model.state_dict(), MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id))

    # Result visualization
    result_visual = ResultVisualization(mode=mode,
                                        architecture=architecture,
                                        pred_step=pred_step)
    saved_model.load_state_dict(torch.load(MODEL_SAVE_PATH % (EXPERIMENT_ID, MODEL_ID, para_id)))
    saved_model.eval()
    output_list, label_list = Helper.basic_predict_test(saved_model, test_loader)
    result_visual.loss_plot(train_loss=train_loss_list, test_loss=test_loss_list)
    for output, label in zip(output_list, label_list):
        result_visual.data_plot(groundtruth=label, prediction=output, zoomInRange=0, zoomInBias=0)

    '''

    # Result Ploting
    output1 = output_list[-1].detach().numpy()
    output1 = output1[0, :, :]
    label1 = label_list[-1].detach().numpy()
    label1 = label1[0, :, :]
    x_axis = np.arange(30)
    epoch_axis = np.arange(epoch)

    result_visual = ResultVisualization(mode=mode,
                                        architecture=architecture,
                                        pred_step=pred_step)

    result_visual.loss_plot(train_loss=train_loss_list, test_loss=test_loss_list)
    result_visual.data_plot(groundtruth=label1, prediction=output1, zoomInRange=0, zoomInBias=0)
    '''






