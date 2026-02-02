import torch.nn as nn
import numpy as np
from modules.vae.surprise_vae import LatentPriorEncoder, LatentPosteriorEncoder, Decoder
import torch as th


class SurpriseAgent(nn.Module):
    def __init__(self, history_input_dim, args):
        super(SurpriseAgent, self).__init__()
        self.args = args
        self.history_input_dim = history_input_dim

        decoder_output_dim = int(np.prod(args.OBS_SHAPE))

        self.latent_prior_encoder = LatentPriorEncoder(history_input_dim, args)
        self.latent_posterior_encoder = LatentPosteriorEncoder(history_input_dim, args)
        self.decoder = Decoder(decoder_output_dim, args)

        self.vae_params = list(self.latent_prior_encoder.parameters()) + list(self.latent_posterior_encoder.parameters()) + list(self.decoder.parameters())
        self.vae_named_parameters = list(self.latent_prior_encoder.named_parameters()) + list(self.latent_posterior_encoder.named_parameters()) + list(self.decoder.named_parameters())
        self._named_children = list(self.latent_prior_encoder.named_children()) + list(self.latent_posterior_encoder.named_children()) + list(self.decoder.named_children())

    def prior_init_hidden(self):
        # make hidden states on same device as model
        hidden = self.latent_prior_encoder.FC_input.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()
        return hidden

    def posterior_init_hidden(self):
        # make hidden states on same device as model
        hidden = self.latent_posterior_encoder.FC_input.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()
        return hidden

    def reparameterization(self, mean, var):
        epsilon = th.randn_like(var).to(self.args.DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def prior_encoder_forward(self, history, hidden=None):
        history = history.reshape(-1, self.history_input_dim)
        # action = action.reshape(-1, self.args.N_ACTIONS)
        mean, log_var, hidden = self.latent_prior_encoder(history, hidden)
        z = self.reparameterization(mean, th.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return mean, log_var, z, hidden

    def posterior_encoder_forward(self, history, action, next_obs, hidden=None):
        history = history.reshape(-1, self.history_input_dim)
        action = action.reshape(-1, self.args.N_ACTIONS)
        next_obs = next_obs.reshape(-1, self.args.OBS_SHAPE)
        mean, log_var, hidden = self.latent_posterior_encoder(history, action, next_obs, hidden)
        z = self.reparameterization(mean, th.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return mean, log_var, z, hidden

    def decoder_forward(self, z):
        x_hat = self.decoder(z)
        return x_hat
