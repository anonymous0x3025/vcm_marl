import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.cnn.cnn_module import CNNModule


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args, output_shape=None):
        super(RNNAgent, self).__init__()
        self.args = args

        if output_shape is None:
            output_shape = args.N_ACTIONS

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state):
        if self.use_cnn_model:
            inputs = self.cnn_module(inputs)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        if self.args.ACTION_SPACE == 'continuous':
            q = F.tanh(q)
        return q, h


# class IntrinsicRNNAgent(nn.Module):
#     def __init__(self, input_shape, args, output_shape=None):
#         super(IntrinsicRNNAgent, self).__init__()
#         self.args = args
#
#         if output_shape is None:
#             output_shape = args.N_ACTIONS
#
#         self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
#         self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
#         self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)
#
#         self.hyper_noise_fc1 = nn.Linear(args.LATENT_DIM, args.RNN_HIDDEN_DIM * args.N_ACTIONS)
#
#         self.latent_fc1 = nn.Linear(args.LATENT_DIM, args.RNN_HIDDEN_DIM)
#         self.latent_fc2 = nn.Linear(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
#         self.latent_fc3 = nn.Linear(args.RNN_HIDDEN_DIM, args.N_ACTIONS)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()
#
#     def forward(self, inputs, hidden_state, latent):
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
#         h = self.rnn(x, h_in)
#         q = self.fc2(h)
#
#         z = F.tanh(self.latent_fc1(latent))
#         z = F.tanh(self.latent_fc2(z))
#         wz = self.latent_fc3(z)
#
#         wq = q * wz
#
#         return wq, h


class IntrinsicRNNAgent(nn.Module):
    def __init__(self, input_shape, args, output_shape=None):
        super(IntrinsicRNNAgent, self).__init__()
        self.args = args

        if output_shape is None:
            output_shape = args.N_ACTIONS

        self.fc1_obs_encoding = nn.Linear(input_shape, args.LATENT_DIM)

        self.fc1_latent_encoding = nn.Linear(args.LATENT_DIM, args.LATENT_DIM)

        input_shape = args.LATENT_DIM + args.LATENT_DIM

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)

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
        if self.args.ACTION_SPACE == 'continuous':
            q = F.tanh(q)
        return q, h


class IntrinsicBeliefRNNAgent(nn.Module):
    def __init__(self, input_shape, args, output_shape=None):
        super(IntrinsicBeliefRNNAgent, self).__init__()
        self.args = args

        if output_shape is None:
            output_shape = args.N_ACTIONS

        self.fc1_obs_encoding = nn.Linear(input_shape, args.LATENT_DIM)

        if self.args.SAMPLE_EMBEDDINGS:
            self.fc1_latent_encoding = nn.Linear(args.LATENT_DIM, args.LATENT_DIM)
        else:
            self.fc1_latent_encoding = nn.Linear(args.LATENT_DIM+args.LATENT_DIM, args.LATENT_DIM)

        input_shape = args.LATENT_DIM + args.LATENT_DIM

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)

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
        if self.args.ACTION_SPACE == 'continuous':
            q = F.tanh(q)
        return q, h


class UtilRNNAgent(nn.Module):
    def __init__(self, input_shape, args, output_shape=None):
        super(UtilRNNAgent, self).__init__()
        self.args = args

        if output_shape is None:
            output_shape = args.N_ACTIONS

        input_shape += args.LATENT_DIM

        self.use_cnn_model = False
        if isinstance(input_shape, tuple):
            self.cnn_module = CNNModule(input_shape=input_shape, output_shape=args.IMAGE_FLATTENED_SIZE)
            input_shape = args.IMAGE_FLATTENED_SIZE
            self.use_cnn_model = True

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state):
        if self.use_cnn_model:
            inputs = self.cnn_module(inputs)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        if self.args.ACTION_SPACE == 'continuous':
            q = F.tanh(q)
        return q, h
