import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, args):
        super(Encoder, self).__init__()
        self.args = args
        hidden_dim = args.VAE_HIDDEN_DIM
        latent_dim = args.LATENT_DIM

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, hidden_state):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_in = hidden_state.reshape(-1, self.args.VAE_HIDDEN_DIM)
        h_out = self.gru(h_, h_in)
        h_ = self.LeakyReLU(self.FC_input2(h_out))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var, h_out


class Decoder(nn.Module):
    def __init__(self, output_dim, args):
        super(Decoder, self).__init__()
        latent_dim = args.LATENT_DIM
        hidden_dim = args.VAE_HIDDEN_DIM

        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        # x_hat = self.FC_output(h)
        x_hat = self.FC_output(h)
        return x_hat


class UtilDecoder(nn.Module):
    def __init__(self, obs_dim, action_dim, input_dim, output_dim, args):
        super(UtilDecoder, self).__init__()
        hidden_dim = args.VAE_HIDDEN_DIM

        self.FC_obs_encoder = nn.Linear(obs_dim, args.OBS_ENCODING_DIM)
        self.FC_action_encoder = nn.Linear(action_dim, args.ACTION_ENCODING_DIM)

        self.FC_hidden = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, obs, actions):
        h_obs = self.LeakyReLU(self.FC_obs_encoder(obs))
        h_actions = self.LeakyReLU(self.FC_action_encoder(actions))
        h = th.cat((x, h_obs, h_actions), dim=-1)

        h = self.LeakyReLU(self.FC_hidden(h))
        h = self.LeakyReLU(self.FC_hidden2(h))
        # x_hat = self.FC_output(h)
        x_hat = self.FC_output(h)
        return x_hat


class IntrinsicDecoder(nn.Module):
    def __init__(self, decoder_obs_input_dim, decoder_input_dim, args):
        super(IntrinsicDecoder, self).__init__()
        self.args = args
        self.hidden_dim = args.VAE_HIDDEN_DIM
        self.n_agents = args.N_AGENTS
        self.obs_dim = int(np.prod(args.OBS_SHAPE))
        self.action_dim = args.N_ACTIONS
        self.input_latent_dim = args.LATENT_DIM

        self.FC_latent_encoder1 = nn.Linear(args.LATENT_DIM, args.LATENT_DIM)
        self.FC_input_encoder1 = nn.Linear(decoder_obs_input_dim, args.OBS_ENCODING_DIM)
        self.FC_action_encoder1 = nn.Linear(self.action_dim, args.ACTION_ENCODING_DIM)

        self.FC_hidden = nn.Linear(decoder_input_dim, self.hidden_dim)
        self.FC_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_output = nn.Linear(self.hidden_dim, self.obs_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, obs, action):
        x = x.reshape(-1, self.input_latent_dim)

        x = self.LeakyReLU(self.FC_latent_encoder1(x))
        h_obs = self.LeakyReLU(self.FC_input_encoder1(obs))
        h_action = self.LeakyReLU(self.FC_action_encoder1(action))

        h = th.cat((x, h_obs, h_action), dim=-1)

        h = self.LeakyReLU(self.FC_hidden(h))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat


class VAEModel(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(VAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder.FC_input.weight.new(1, self.args.VAE_HIDDEN_DIM).zero_()

    def reparameterization(self, mean, var):
        epsilon = th.randn_like(var).to(self.args.DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x, hidden_state):
        mean, log_var, hidden_state = self.encoder(x, hidden_state)
        z = self.reparameterization(mean, th.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        return mean, log_var, z, x_hat, hidden_state
