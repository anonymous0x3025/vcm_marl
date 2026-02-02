import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class ICESCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(ICESCritic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.RNN_HIDDEN_DIM)
        self.rnn = nn.GRUCell(args.RNN_HIDDEN_DIM, args.RNN_HIDDEN_DIM)
        self.fc2 = nn.Linear(args.RNN_HIDDEN_DIM, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.RNN_HIDDEN_DIM)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.RNN_HIDDEN_DIM).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.RNN_HIDDEN_DIM)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)
