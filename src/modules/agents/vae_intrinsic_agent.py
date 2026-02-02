import torch.nn as nn
import numpy as np
from modules.vae.vae import Encoder, IntrinsicDecoder
import torch as th


class VAEIntrinsicAgent(nn.Module):
    def __init__(self, encoder_input_shape, decoder_input_dim, args):
        super(VAEIntrinsicAgent, self).__init__()
        self.args = args
        self.encoder_input_shape = encoder_input_shape
        self.decoder_obs_input_dim = encoder_input_shape
        self.decoder_input_dim = decoder_input_dim

        self.encoder = Encoder(self.encoder_input_shape, args)
        self.decoder = IntrinsicDecoder(self.decoder_obs_input_dim, self.decoder_input_dim, args)
        self.vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.vae_named_parameters = list(self.encoder.named_parameters()) + list(self.decoder.named_parameters())
        self._named_children = list(self.encoder.named_children()) + list(self.decoder.named_children())

    def init_hidden(self):
        # make hidden states on same device as model
        hidden = self.encoder.FC_input.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()
        return hidden

    def reparameterization(self, mean, var):
        epsilon = th.randn_like(var).to(self.args.DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encoder_forward(self, inputs, hidden=None):
        inputs = inputs.reshape(-1, self.encoder_input_shape)
        mean, log_var, hidden = self.encoder(inputs, hidden)
        z = self.reparameterization(mean, th.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        if self.args.SAMPLE_EMBEDDINGS:
            z_for_policy = z
        else:
            z_for_policy = th.cat((mean, log_var), dim=-1)
        return mean, log_var, z, z_for_policy, hidden

    def decoder_forward(self, z, obs, actions_onehot):
        x_hat = self.decoder(z, obs, actions_onehot)
        return x_hat
