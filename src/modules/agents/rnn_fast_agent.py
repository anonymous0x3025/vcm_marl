import torch.nn as nn
import torch.nn.functional as F


class RNNFastAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNFastAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.RNN_HIDDEN_DIM,
            num_layers=1,
            hidden_size=args.RNN_HIDDEN_DIM,
            batch_first=True
        )
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, args.N_ACTIONS)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state):
        bs = inputs.shape[0]
        epi_len = inputs.shape[1]
        num_feat = inputs.shape[2]
        inputs = inputs.reshape(bs * epi_len, num_feat)
        x = F.relu(self.fc1(inputs))
        x = x.reshape(bs, epi_len, self.args.RNN_HIDDEN_DIM)
        h_in = hidden_state.reshape(1, bs, self.args.RNN_HIDDEN_DIM).contiguous()
        x, h = self.rnn(x, h_in)
        x = x.reshape(bs * epi_len, self.args.RNN_HIDDEN_DIM)
        q = self.fc2(x)
        q = q.reshape(bs, epi_len, self.args.N_ACTIONS)
        return q, h
