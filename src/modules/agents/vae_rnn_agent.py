import copy

import torch.nn as nn
import numpy as np
from modules.cnn.cnn_module import CNNModule
from modules.vae.vae import Encoder, UtilDecoder
import torch as th


class VAERNNAgent(nn.Module):
    def __init__(self, encoder_input_shape, obs_dim, action_dim, decoder_input_shape, decoder_output_shape, args, output_shape=None):
        super(VAERNNAgent, self).__init__()
        self.args = args
        self.encoder_input_shape = encoder_input_shape
        self.decoder_input_shape = decoder_input_shape
        self.decoder_output_shape = decoder_output_shape

        if output_shape is None:
            output_shape = args.N_ACTIONS

        self.use_cnn_model = False
        if isinstance(encoder_input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=encoder_input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            self.reshaped_input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.encoder = Encoder(self.encoder_input_shape, args)
        self.decoder = UtilDecoder(obs_dim, action_dim, self.decoder_input_shape, self.decoder_output_shape, args)
        self.vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)
        self.util_params = list(self.rnn.parameters()) + list(self.fc.parameters())

        self.target_rnn = copy.deepcopy(self.rnn)
        self.target_fc = copy.deepcopy(self.fc)
        self.target_util_params = list(self.target_rnn.parameters()) + list(self.target_fc.parameters())

    def init_hidden(self):
        # make hidden states on same device as model
        hidden = self.encoder.FC_mean.weight.new(1, self.args.LATENT_DIM).zero_()
        util_hidden = self.encoder.FC_mean.weight.new(1, self.args.LATENT_DIM).zero_()
        return hidden, util_hidden

    def reparameterization(self, mean, var):
        epsilon = th.randn_like(var).to(self.args.DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encoder_forward(self, inputs, hidden=None):
        if self.use_cnn_model:
            inputs = np.reshape(inputs, (-1, *self.encoder_input_shape))
            inputs = self.cnn_module(inputs)
            inputs = inputs.reshape(-1, self.reshaped_input_shape)
        else:
            inputs = inputs.reshape(-1, self.encoder_input_shape)

        mean, log_var, hidden = self.encoder(inputs, hidden)
        z = self.reparameterization(mean, th.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return mean, log_var, z, hidden

    def decoder_forward(self, z, obs, actions_onehot):
        x_hat = self.decoder(z, obs, actions_onehot)
        return x_hat

    def util_forward(self, z, util_hidden=None, target=False):
        util_hidden = util_hidden.reshape(-1, self.args.LATENT_DIM)
        if not target:
            util_hidden = self.rnn(z, util_hidden)
            util_q = self.fc(util_hidden)
        else:
            util_hidden = self.target_rnn(z, util_hidden)
            util_q = self.target_fc(util_hidden)
        return util_q, util_hidden

    def load_state(self):
        self.rnn.load_state_dict(self.target_rnn.state_dict())
        self.fc.load_state_dict(self.target_fc.state_dict())
