import torch.nn as nn
import torch


class Basic_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout, pred_step):
        super(Basic_GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          dropout=dropout)
        self.pred_step = pred_step
        self.length_extender = nn.Linear(1, hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, pred_step)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.feature_compressor = nn.Linear(hidden_dim//2, input_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, input):
        output, _ = self.gru(input)
        context_vec = output[:, -1, :]
        context_vec = context_vec[:, None]
        temp = torch.permute(context_vec, (0, 2, 1))
        temp_extended = self.length_extender(temp)
        temp_extended = self.activation(temp_extended)
        temp_extended = self.dense1(temp_extended)
        temp_extended = self.activation(temp_extended)
        extended_context = torch.permute(temp_extended, (0, 2, 1))
        extended_context = self.dense2(extended_context)
        extended_context = self.activation(extended_context)
        output = self.feature_compressor(extended_context)

        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Encoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)
        # self.mlp = Basic_MLP(in_channels=input_dim, hidden_channels=hidden_dim, output_channels=32)
        # self.num_layers = num_layers
        # self.hidden_dim = hidden_dim
        # self.batch_size = batch_size

    def forward(self, inputs):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # h_0 = torch.randn((self.num_layers, inputs.shape[0], self.hidden_dim)).to(device)
        # c_0 = torch.randn(self.num_layers, 4, self.hidden_dim).to(device)
        # inputs = self.mlp(inputs)
        enc_output, enc_state = self.encoder(inputs)
        return enc_output, enc_state


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)
        # self.mlp1 = Basic_MLP(in_channels=input_dim, hidden_channels=hidden_dim, output_channels=32)
        # self.mlp2 = Basic_MLP(in_channels=hidden_dim, hidden_channels=hidden_dim, output_channels=input_dim)
        self.dense = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs, state):
        # inputs = self.mlp1(inputs)
        outputs, dec_state = self.gru(inputs, state)
        # outputs = self.tanh(output)
        # pred = self.relu(outputs)
        pred = self.dense(outputs)

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

class Basic_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels):
        super(Basic_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, output_channels)
        )

    def forward(self, input):
        output = self.mlp(input)
        return output


