import torch as th
import torch.nn as nn


class IntrinsicValueCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(IntrinsicValueCritic, self).__init__()
        self.args = args

        self.fc1_obs_encoding = nn.Linear(input_shape, args.LATENT_DIM)

        self.fc1_latent_encoding = nn.Linear(args.LATENT_DIM, args.LATENT_DIM)

        input_shape = args.LATENT_DIM + args.LATENT_DIM

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, 1)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state, latent):
        x = self.LeakyReLU(self.fc1_obs_encoding(inputs))

        latent = self.LeakyReLU(self.fc1_latent_encoding(latent))

        x = th.cat([x, latent], dim=1)

        x = self.LeakyReLU(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
