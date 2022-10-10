import torch
import random


class ModelTrainer:

    def __init__(self, model, loss_func, optimizer, device, pred_step):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.pred_step = pred_step

    def basic_train(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output = self.model(inputs)
            loss = self.loss_func(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def basic_predict(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output = self.model(inputs)
            test_loss += self.loss_func(output, labels).item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    def enc_dec_train(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for encoder_inputs, (decoder_inputs, labels) in train_loader:
            encoder_inputs, decoder_inputs, labels = encoder_inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            teach_enforce = random.random()
            if teach_enforce >= 0.5:
                enc_output, enc_state = self.model.encoder(encoder_inputs)
                dec_output, dec_state = self.model.decoder(encoder_inputs[:, -2, None], enc_state)
                dec_pred = dec_output
                for _ in range(self.pred_step - 1):
                    dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                    dec_output = torch.cat((dec_output, dec_pred), 1)
            else:
                dec_output, _ = self.model(encoder_inputs, decoder_inputs)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def enc_dec_predict(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for encoder_inputs, (decoder_input, labels) in test_loader:
            encoder_inputs, decoder_input, labels = encoder_inputs.to(self.device), decoder_input.to(
                self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs)
            dec_output, dec_state = self.model.decoder(encoder_inputs[:, -2, None], enc_state)
            dec_pred = dec_output
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss