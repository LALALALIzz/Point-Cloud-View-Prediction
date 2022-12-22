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
        for encoder_inputs, (_, decoder_inputs, labels) in train_loader:
            encoder_inputs, decoder_inputs, labels = encoder_inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            teach_enforce = 0.7
            if teach_enforce >= 0.5:
                enc_output, enc_state = self.model.encoder(encoder_inputs)
                dec_output, dec_state = self.model.decoder(encoder_inputs[:, -1, None], enc_state)
                dec_pred = dec_output.clone()
                for _ in range(self.pred_step - 1):
                    dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                    dec_output = torch.cat((dec_output, dec_pred), 1)
            else:
                dec_output, _ = self.model(encoder_inputs, decoder_inputs)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def enc_dec_train2(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        enc_outputs = None
        for encoder_inputs, (decoder_inputs, labels) in train_loader:
            encoder_inputs, decoder_inputs, labels = encoder_inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            h_0 = torch.normal(0, 1, (self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            teach_enforce = random.random()
            if teach_enforce >= 0.5:
                enc_outputs, enc_state = self.model.encoder(encoder_inputs, h_0)
                dec_output, dec_state = self.model.decoder(decoder_inputs[:, 0, None], enc_state)
                dec_pred = dec_output
                for _ in range(self.pred_step - 1):
                    dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                    dec_output = torch.cat((dec_output, dec_pred), 1)
            else:
                enc_outputs, dec_output, _ = self.model(encoder_inputs, h_0, decoder_inputs)
            output = torch.concat((enc_outputs, dec_output), dim=1)
            label = torch.concat((encoder_inputs, labels), dim=1)
            loss = self.loss_func(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def enc_dec_train3(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        enc_outputs = None
        for encoder_inputs, (decoder_inputs, labels) in train_loader:
            encoder_inputs, decoder_inputs, labels = encoder_inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            h_0 = torch.normal(0, 1, (self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            teach_enforce = 0.6
            if teach_enforce >= 0.5:
                enc_outputs, enc_state = self.model.encoder(encoder_inputs, h_0)
                start_token = torch.normal(0, 1, size=(encoder_inputs.shape[0], 1, self.hidden_dim)).to(self.device)
                dec_output, dec_state = self.model.decoder(start_token, enc_state)
                dec_pred = dec_output.clone()
                for _ in range(self.pred_step - 1):
                    dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                    dec_output = torch.cat((dec_output, dec_pred), 1)
            else:
                enc_outputs, dec_output, _ = self.model(encoder_inputs, h_0, decoder_inputs)
            output = self.model.output(torch.concat((enc_outputs, dec_output), dim=1))
            label = torch.concat((encoder_inputs, labels), dim=1)
            loss = self.loss_func(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    # Week 8
    # Experiment 14
    def enc_dec_train4(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for encoder_inputs, (start_token, labels) in train_loader:
            encoder_inputs, start_token, labels = encoder_inputs.to(self.device), start_token.to(self.device), labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
            dec_output, dec_state = self.model.decoder(start_token[:, 0, :].unsqueeze(dim=1), enc_state)
            dec_pred = dec_output.clone()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def encoder_train(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, (enc_labels, _, _) in train_loader:
            inputs, enc_labels = inputs.to(self.device), enc_labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            output, _ = self.model(inputs)#, h_0)
            loss = self.loss_func(output, enc_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def encoder_validation(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for inputs, (enc_labels, _, _) in test_loader:
            inputs, labels = inputs.to(self.device), enc_labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            output, _ = self.model(inputs)#, h_0)
            loss = self.loss_func(output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    def decoder_train(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, (_, dec_inputs, labels) in train_loader:
            inputs, dec_inputs, labels = inputs.to(self.device), dec_inputs.to(self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(inputs)#, h_0)
            dec_output, dec_state = self.model.decoder(dec_inputs[:, 0, :].unsqueeze(dim=1), enc_state)
            dec_pred = dec_output.clone()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    def decoder_train2(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, (_, decoder_inputs, labels) in train_loader:
            inputs, decoder_inputs, labels = inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(inputs)
            dec_output, dec_state = self.model.decoder(decoder_inputs, enc_state)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    # Week 9
    def decoder_train3(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, (_, _, labels) in train_loader:
            decoder_inputs = torch.mean(input=inputs, dim=1, keepdim=True)
            decoder_inputs = decoder_inputs.repeat(1, labels.shape[1], 1)
            inputs, decoder_inputs, labels = inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(inputs)
            dec_output, dec_state = self.model.decoder(decoder_inputs, enc_state)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    # Week 9
    def decoder_train4(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, (_, _, labels) in train_loader:
            decoder_inputs = inputs[:, -1, :].unsqueeze(dim=1)
            decoder_inputs = decoder_inputs.repeat(1, inputs.shape[1], 1)
            inputs, decoder_inputs, labels = inputs.to(self.device), decoder_inputs.to(self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(inputs)
            dec_output, dec_state = self.model.decoder(decoder_inputs, enc_state)
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss

    # Week 10
    def decoder_train5(self, train_loader):
        self.model.train()
        train_loss = 0
        counter = 0
        for inputs, (_, dec_inputs, labels) in train_loader:
            inputs, dec_inputs, labels = inputs.to(self.device), dec_inputs.to(self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(inputs)
            dec_input = inputs[:, -1, :].detach().unsqueeze(dim=1)
            dec_output, dec_state = self.model.decoder(dec_input, enc_state)
            dec_pred = dec_output.clone().detach()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
                dec_pred = dec_pred.clone().detach()
            loss = self.loss_func(dec_output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            train_loss += loss.item()
            counter += 1
        train_loss = train_loss / counter
        return train_loss



    def enc_dec_predict(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for encoder_inputs, (_, decoder_input, labels) in test_loader:
            encoder_inputs, decoder_input, labels = encoder_inputs.to(self.device), decoder_input.to(
                self.device), labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs)#, h_0)
            dec_output, dec_state = self.model.decoder(encoder_inputs[:, -1, None], enc_state)
            dec_pred = dec_output.clone()
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
            h_0 = torch.normal(0, 1, (self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
            start_token = torch.normal(0, 1, size=(encoder_inputs.shape[0], 1, self.hidden_dim)).to(self.device)
            dec_output, dec_state = self.model.decoder(start_token, enc_state)
            dec_pred = dec_output.clone()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            dec_output = self.model.output(dec_output)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    def enc_dec_predict3(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for encoder_inputs, (start_token, labels) in test_loader:
            encoder_inputs, start_token, labels = encoder_inputs.to(self.device), start_token.to(self.device), labels.to(self.device)
            h_0 = torch.zeros((self.num_layers, encoder_inputs.shape[0], self.hidden_dim))
            h_0 = h_0.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs, h_0)
            dec_output, dec_state = self.model.decoder(start_token[:, 0, :].unsqueeze(dim=1), enc_state)
            dec_pred = dec_output.clone()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, dec_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    def enc_dec_predict4(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for encoder_inputs, (_, _, labels) in test_loader:
            encoder_inputs, labels = encoder_inputs.to(self.device), labels.to(self.device)
            enc_output, enc_state = self.model.encoder(encoder_inputs)
            dec_output, dec_state = self.model.decoder(encoder_inputs[:, -1, None], enc_state)
            dec_pred = dec_output.clone()
            for _ in range(self.pred_step - 1):
                dec_pred, dec_state = self.model.decoder(dec_pred, enc_state)
                dec_output = torch.cat((dec_output, dec_pred), 1)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    # Week 9
    def enc_dec_predict5(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for inputs, (_, _, labels) in test_loader:
            decoder_inputs = torch.mean(input=inputs, dim=1, keepdim=True)
            decoder_inputs = decoder_inputs.repeat(1, labels.shape[1], 1)
            inputs, decoder_inputs, labels = inputs.to(self.device), decoder_inputs.to(self.device), labels.to(
                self.device)
            enc_output, enc_state = self.model.encoder(inputs)
            dec_output, dec_state = self.model.decoder(decoder_inputs, enc_state)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss

    # Week 9
    def enc_dec_predict6(self, test_loader):
        self.model.eval()
        test_loss = 0
        counter = 0
        for inputs, (_, _, labels) in test_loader:
            decoder_inputs = inputs[:, -1, :].unsqueeze(dim=1)
            decoder_inputs = decoder_inputs.repeat(1, inputs.shape[1], 1)
            inputs, decoder_inputs, labels = inputs.to(self.device), decoder_inputs.to(self.device), labels.to(
                self.device)
            enc_output, enc_state = self.model.encoder(inputs)
            dec_output, dec_state = self.model.decoder(decoder_inputs, enc_state)
            loss = self.loss_func(dec_output, labels)
            test_loss += loss.item()
            counter += 1
        test_loss = test_loss / counter
        return test_loss