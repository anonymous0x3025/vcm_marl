import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.n_agents = args.N_AGENTS

        self.latent_dim = args.LATENT_DIM

        self.h = args.N_HEADS
        self.d_model = args.MODEL_DIM
        self.d_k = args.MODEL_DIM // args.N_HEADS

        self.embedding = nn.Linear(self.latent_dim, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        self.out_linear = nn.Linear(self.d_model, self.latent_dim)
        nn.init.xavier_normal_(self.out_linear.weight)

        self.dropout = nn.Dropout(0.1)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        sequence_dim = x.shape[1]
        x = x.reshape(-1, self.latent_dim)

        # Multi-Head Self Attention
        h = self.embedding(x)
        h = self.layer_norm(h)
        h = h.view(-1, self.d_model)

        q_linear = self.LeakyReLU(self.q_linear(h).view(-1, self.n_agents, self.h, self.d_k))
        k_linear = self.LeakyReLU(self.k_linear(h).view(-1, self.n_agents, self.h, self.d_k))
        v_linear = self.LeakyReLU(self.v_linear(h).view(-1, self.n_agents, self.h, self.d_k))

        q_linear = q_linear.transpose(1, 2)
        k_linear = k_linear.transpose(1, 2)
        v_linear = v_linear.transpose(1, 2)

        k_linear = k_linear.transpose(-2, -1)

        score = th.matmul(q_linear, k_linear)
        scaled_score = score / np.sqrt(self.d_k)

        weight = F.softmax(scaled_score, dim=-1)
        attention_out = th.matmul(weight, v_linear)

        attention_out = attention_out.transpose(1, 2).contiguous().view(-1, self.n_agents, self.d_model)
        attention_out = self.dropout(self.LeakyReLU(self.out_linear(attention_out)))
        attention_out = attention_out.view(-1, sequence_dim, self.latent_dim)

        return attention_out