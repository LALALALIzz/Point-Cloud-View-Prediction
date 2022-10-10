import torch
import random


class ModelTrainer:

    def __init__(self, model, loss_func, optimizer, device, pred_step, num_layers, hidden_dim):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device
        self.pred_step = pred_step
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim


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
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            teach_enforce = 0.1
            if teach_enforce >= 0.5:
                enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
                dec_output, dec_state = self.model.decoder(encoder_inputs[:, -1, None], enc_state)
                dec_pred = dec_output.clone()
                for _ in range(self.pred_step - 1):
                    dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                    dec_output = torch.cat((dec_output, dec_pred), 1)
            else:
                dec_output, _ = self.model(encoder_inputs, h_0, decoder_inputs)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def enc_dec_train2(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for encoder_inputs, (decoder_inputs, labels) in train_loader:
            encoder_inputs, decoder_inputs, labels = encoder_inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            h_0_com = torch.zeros((1, encoder_inputs.shape[0], 3))
            h_0_com = h_0_com.to(self.device)
            teach_enforce = 0.1
            if teach_enforce >= 0.5:
                enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
                dec_output, dec_state = self.model.decoder(encoder_inputs[:, -1, None], enc_state)
                dec_pred = dec_output.clone()
                for _ in range(self.pred_step - 1):
                    dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                    dec_output = torch.cat((dec_output, dec_pred), 1)
            else:
                dec_output = self.model(encoder_inputs, h_0, decoder_inputs)
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
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
            dec_output, dec_state = self.model.decoder(decoder_input[:, 0, None], enc_state)
            dec_pred = dec_output
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    def enc_dec_predict2(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for encoder_inputs, (decoder_input, labels) in test_loader:
            encoder_inputs, decoder_input, labels = encoder_inputs.to(self.device), decoder_input.to(
                self.device), labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
            dec_output, dec_state = self.model.decoder(encoder_inputs[:, -1, None], enc_state)
            dec_output = self.model.compressor(dec_output)
            dec_pred = dec_output.clone()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_pred = self.model.compressor(dec_pred)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss