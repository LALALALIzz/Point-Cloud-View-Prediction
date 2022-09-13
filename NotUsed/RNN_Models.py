import numpy as np
import torch
import torch.nn as nn

'''
class RNN_Encoder_Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_step, num_layers):
        super(RNN_Encoder_Decoder, self).__init__()
        self.out_step = out_step
        self.layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 6)

    def forward(self, inputs):
        encoder_output, (h_e, c_e) = self.encoder(inputs)
        repeated_he = h_e.repeat(self.out_step, 1, 1)
        repeated_he = torch.permute(repeated_he, (1, 0, 2))
        decoder_output, (h_d, c_d) = self.decoder(repeated_he, (h_e, c_e))
        prediction = self.dense(decoder_output)
        return prediction


class RNN_Plain(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_step, num_layers):
        super(RNN_Plain, self).__init__()
        self.out_step = out_step
        self.layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(in_dim, hidden_dim, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim, 6)

    def forward(self, inputs):
        output, h_n = self.rnn(inputs)
        repeated_hn = h_n.repeat(self.out_step, 1, 1)
        repeated_hn = torch.permute(repeated_hn, (1, 0, 2))
        prediction = self.dense(repeated_hn)
        return prediction
'''


class BaselineGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, drop_out):
        super(BaselineGRU, self).__init__()
        self.model = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True, dropout=drop_out)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(hidden_dim, in_dim)

    def forward(self, inputs):
        outputs, _ = self.model(inputs)
        outputs = self.tanh(outputs)
        pred = self.dense(outputs)
        pred = self.tanh(pred)
        return pred


class BaselineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BaselineLinear, self).__init__()
        self.model = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        pred = self.model(inputs)
        return pred


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, drop_out):
        super(Encoder, self).__init__()
        self.encoder = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True, dropout=drop_out)

    def forward(self, inputs):
        enc_output, enc_state = self.encoder(inputs)
        return enc_output, enc_state


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, drop_out):
        super(Decoder, self).__init__()
        self.decoder = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True, dropout=drop_out)
        self.dense = nn.Linear(hidden_dim, in_dim)
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()

    def forward(self, inputs, state):
        outputs, dec_state = self.decoder(inputs, state)
        #outputs = self.tanh(output)
        pred = self.dense(outputs)
        pred = self.tanh(pred)
        return pred, dec_state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        _, enc_state = self.encoder(encoder_input)
        pred, dec_state = self.decoder(decoder_input, enc_state)

        return pred, dec_state



class TestModel(nn.Module):
    def __init__(self, num_features, d_model, num_layers, num_head, observ_step, pred_step):
        super(TestModel, self).__init__()
        self.embedding_layer = nn.Linear(num_features, d_model)
        self.compressor = nn.Linear(d_model, num_features)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(observ_step, pred_step)

    def forward(self, input):
        embedded_input = self.embedding_layer(input)
        encoder_output = self.encoder(embedded_input)
        densed_output = self.compressor(encoder_output)
        changed_output = torch.permute(densed_output, (0, 2, 1))
        dense_output = self.dense(changed_output)
        output = self.tanh(torch.permute(self.tanh(dense_output), (0, 2, 1)))

        return output

class Transformer(nn.Module):
    def __init__(self, num_features, num_layers, num_head, observ_step, pred_step):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(num_features, num_head)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.dense = nn.Linear(observ_step, pred_step)

    def forward(self, input):
        encoder_output = self.encoder(input)
        changed_output = torch.permute(encoder_output, (0, 2, 1))
        dense_output = self.dense(changed_output)
        output = torch.permute(dense_output, (0, 2, 1))

        return output

class AlteredGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, observ_step, pred_step):
        super(AlteredGRU, self).__init__()
        self.model = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        self.tanh = nn.Tanh()
        self.dense1 = nn.Linear(observ_step, pred_step)
        self.dense2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, inputs):
        outputs, _ = self.model(inputs)
        outputs = self.tanh(outputs)
        changed_output = torch.permute(outputs, (0, 2, 1))
        densed_output = self.dense1(changed_output)
        dens1out = torch.permute(densed_output, (0, 2, 1))
        dens1out = self.tanh(dens1out)
        pred = self.dense2(dens1out)
        pred = self.tanh(pred)

        return pred






