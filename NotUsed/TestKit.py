import torch
import torch.nn as nn
import torch.optim as optim
from pipeline import Pipeline
from Transformer_EN import Transformer

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    datasetName = "NJIT"
    time_step = 250
    pred_step = 250
    batch_size = 5
    num_feature = 6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = "angular"
    pipeline_test = Pipeline(
        datasetName=datasetName,
        time_step=time_step,
        pred_step=pred_step,
        num_feature=num_feature,
        batch_size=batch_size,
        random=1,
        device=device,
        mode=mode)

    train_loader, test_loader = pipeline_test.Data_iter()
    #num_hiddens = 90
    num_layers = 1
    learn_rate = 0.00001
    d_model = 512
    num_head = 6
    model = Transformer(num_features=num_feature, num_layers=num_layers, num_head=num_head, observ_step=time_step, pred_step=pred_step)
    model = model.double()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_func = nn.MSELoss()

    #pipeline_test.automation(epoch=500, model=model, optimizer=optimizer, loss_func=loss_func, train_loader=train_loader, test_loader=test_loader, patience=5)

    '''
    train_loss = []
    test_loss = []
    epoch_num = 500

    for epoch in tqdm(range(1, epoch_num+1)):
        train_loss.append(pipeline_test.train(model=model, train_loader=train_loader,
                                              loss_func=loss_func, optimizer=optimizer).to('cpu').detach().numpy())
        test_loss.append(pipeline_test.predict(model=model, test_loader=test_loader))

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    epoch_list = np.arange(1, epoch_num+1)
    plt.figure()
    plt.plot(epoch_list, train_loss, 'b')
    plt.plot(epoch_list, test_loss, 'r')
    plt.show()
    '''
    model = torch.load('../CHECKPOINTS/GRU_NJIT_00001_500.pt')
    model.to(device)
    print(pipeline_test.predict(model=model, test_loader=test_loader))

