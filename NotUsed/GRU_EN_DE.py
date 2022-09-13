import torch
import torch.nn as nn

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
        output, dec_state = self.decoder(inputs, state)
        outputs = self.tanh(output)
        pred = self.dense(outputs)
        #pred = self.tanh(pred)
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