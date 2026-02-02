import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.n_agents = args.N_AGENTS
        self.embed_dim = args.MIXING_EMBED_DIM
        self.input_dim = self.state_dim = int(np.prod(args.STATE_SHAPE))

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        
        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.HYPERNET_EMBED),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.HYPERNET_EMBED, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.HYPERNET_EMBED),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.HYPERNET_EMBED, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(states).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
            
        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
        
        return y.view(b, t, -1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        
