import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LatentPriorEncoder(nn.Module):
    def __init__(self, history_input_dim, args):
        super(LatentPriorEncoder, self).__init__()
        self.args = args
        hidden_dim = args.VAE_HIDDEN_DIM
        latent_dim = args.LATENT_DIM

        self.FC_input = nn.Linear(history_input_dim, hidden_dim)
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

    def forward(self, history, hidden_state):
        h_ = self.LeakyReLU(self.FC_input(history))
        h_in = hidden_state.reshape(-1, self.args.VAE_HIDDEN_DIM)
        h_out = self.gru(h_, h_in)
        h_ = self.LeakyReLU(self.FC_input2(h_out))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var, h_out


class LatentPosteriorEncoder(nn.Module):
    def __init__(self, history_input_dim, args):
        super(LatentPosteriorEncoder, self).__init__()
        self.args = args
        self.hidden_dim = args.VAE_HIDDEN_DIM
        latent_dim = args.LATENT_DIM

        obs_dim = int(np.prod(args.OBS_SHAPE))
        action_dim = int(np.prod(args.N_ACTIONS))

        self.FC_input_encoder = nn.Linear(history_input_dim, args.HISTORY_ENCODING_DIM)
        self.FC_action_encoder = nn.Linear(action_dim, args.ACTION_ENCODING_DIM)
        self.FC_obs_encoder = nn.Linear(obs_dim, args.OBS_ENCODING_DIM)

        input_dim = args.HISTORY_ENCODING_DIM + args.OBS_ENCODING_DIM + args.ACTION_ENCODING_DIM

        self.h = args.N_HEADS
        self.d_model = args.MODEL_DIM
        self.d_k = args.MODEL_DIM // args.N_HEADS

        self.embedding = nn.Linear(input_dim, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        self.out_linear = nn.Linear(self.d_model, self.hidden_dim)
        nn.init.xavier_normal_(self.out_linear.weight)

        self.dropout = nn.Dropout(0.1)

        self.FC_input = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.FC_input2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_mean = nn.Linear(self.hidden_dim, latent_dim)
        self.FC_var = nn.Linear(self.hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, history, action, next_obs, hidden_state):
        h_history = self.FC_input_encoder(history)
        h_action = self.FC_action_encoder(action)
        h_next_obs = self.FC_obs_encoder(next_obs)

        h = th.cat((h_history, h_action, h_next_obs), dim=-1)

        # Multi-Head Self Attention
        h = self.embedding(h)
        h = self.layer_norm(h)
        h = h.view(-1, self.d_model)

        q_linear = self.LeakyReLU(self.q_linear(h).view(-1, self.args.N_AGENTS, self.h, self.d_k))
        k_linear = self.LeakyReLU(self.k_linear(h).view(-1, self.args.N_AGENTS, self.h, self.d_k))
        v_linear = self.LeakyReLU(self.v_linear(h).view(-1, self.args.N_AGENTS, self.h, self.d_k))

        q_linear = q_linear.transpose(1, 2)
        k_linear = k_linear.transpose(1, 2)
        v_linear = v_linear.transpose(1, 2)

        k_linear = k_linear.transpose(-2, -1)

        score = th.matmul(q_linear, k_linear)
        scaled_score = score / np.sqrt(self.d_k)

        weight = F.softmax(scaled_score, dim=-1)
        attention_out = th.matmul(weight, v_linear)

        attention_out = attention_out.transpose(1, 2).contiguous().view(-1, self.args.N_AGENTS, self.d_model)
        attention_out = self.dropout(self.LeakyReLU(self.out_linear(attention_out)))
        attention_out = attention_out.view(-1, self.hidden_dim)

        # Encoder
        h_ = self.LeakyReLU(self.FC_input(attention_out))
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
        x_hat = self.FC_output(h)
        return x_hat
