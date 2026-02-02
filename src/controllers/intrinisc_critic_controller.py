from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class IntrinsicCriticMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.N_AGENTS
        self.args = args
        self.scheme = scheme
        self.input_shape = self._get_input_shape()
        self._build_agents()

        self.epsilon = args.EPSILON_START

        self.hidden_states = None

        self.device = th.device(self.args.DEVICE)

    def intrinsic_forward(self, z_outs, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, z_outs)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def get_hidden(self):
        return self.hidden_states

    def parameters(self):
        return self.agent.parameters()

    def named_parameters(self):
        return self.agent.named_parameters()

    def named_children(self):
        return self.agent.named_children()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def to_device(self):
        self.agent.to(device=self.device)

    def save_models(self, path, model_name="{}/agent.th"):
        th.save(self.agent.state_dict(), model_name.format(path))

    def save_models_for_hidden(self, path, model_name="{}/hidden_agent.th"):
        th.save(self.agent.state_dict(), model_name.format(path))

    def load_models(self, path, model_name="{}/agent.th"):
        self.agent.load_state_dict(th.load(model_name.format(path), map_location=lambda storage, loc: storage))

    def load_models_for_hidden(self, path, model_name="{}/hidden_agent.th"):
        self.agent.load_state_dict(th.load(model_name.format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent = agent_REGISTRY[self.args.INTRINSIC_AGENT](self.input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.OBS_LAST_ACTION:
            if self.args.ACTION_SPACE == "discrete":
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t-1])
            elif self.args.ACTION_SPACE == "continuous":
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t-1])
            else:
                raise Exception("ACTION SPACE must be defined ! ")
        if self.args.OBS_AGENT_ID:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if isinstance(self.input_shape, int):
            inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        elif isinstance(self.input_shape, tuple):
            inputs = th.cat([x.reshape(bs*self.n_agents, *self.input_shape) for x in inputs], dim=1)
        else:
            raise NotImplementedError
        return inputs

    def _get_input_shape(self):
        input_shape = self.scheme["obs"]["vshape"]
        if self.args.OBS_LAST_ACTION:
            if self.args.ACTION_SPACE == "discrete":
                input_shape += self.scheme["actions_onehot"]["vshape"][0]
            elif self.args.ACTION_SPACE == "continuous":
                input_shape += self.scheme["actions"]["vshape"]
            else:
                raise Exception("ACTION SPACE must be defined ! ")
        if self.args.OBS_AGENT_ID:
            input_shape += self.n_agents

        return input_shape
