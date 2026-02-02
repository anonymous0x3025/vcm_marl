from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
import torch.nn.functional as F
import numpy as np


# This multi-agent controller shares parameters between agents
class SurpriseMAC:
    def __init__(self, scheme, args):
        self.n_agents = args.N_AGENTS
        self.args = args
        self.scheme = scheme
        self.history_input_shape = self._get_input_shape()
        self._build_agents()
        self.agent_output_type = args.AGENT_OUTPUT_TYPE

        self.prior_encoder_hidden = None
        self.posterior_encoder_hidden = None

        self.device = th.device(self.args.DEVICE)

    def prior_encoder_forward(self, ep_batch, t):
        # VAE forwarding to make intrinsic reward
        encoder_inputs = self._build_inputs(ep_batch, t)

        # batch_size = ep_batch.batch_size

        # actions_onehot = ep_batch["actions_onehot"][:, :-1]
        # actions_onehot = actions_onehot.reshape(batch_size * self.n_agents * (ep_batch.max_seq_length - 1), -1)

        mean, log_var, z, self.prior_encoder_hidden = self.exp_vae.prior_encoder_forward(encoder_inputs, self.prior_encoder_hidden)
        return mean, log_var, z

    def posterior_encoder_forward(self, ep_batch, t):
        # VAE forwarding to make intrinsic reward
        encoder_inputs = self._build_inputs(ep_batch, t)

        actions_onehot = ep_batch["actions_onehot"][:, t]
        actions_onehot = actions_onehot.reshape(- 1, self.args.N_ACTIONS)

        next_obs = ep_batch["obs"][:, t]
        next_obs = next_obs.reshape(-1, self.args.OBS_SHAPE)

        mean, log_var, z, self.posterior_encoder_hidden = self.exp_vae.posterior_encoder_forward(encoder_inputs, actions_onehot, next_obs, self.posterior_encoder_hidden)
        return mean, log_var, z
    
    def decoder_forward(self, batch, prior_mean_out, posterior_mean_out, prior_log_var_out, posterior_log_var_out, posterior_z_out):
        batch_size = batch.batch_size

        x_hat_out = self.exp_vae.decoder_forward(posterior_z_out)
        x_hat_out = x_hat_out.reshape(batch_size, batch.max_seq_length - 1, self.n_agents, -1)

        next_obs = batch["obs"][:, 1:]

        # reconstruction loss
        production_error = (x_hat_out - next_obs).pow(2).sum(-1)

        # KLD
        gauss_dim = prior_mean_out.shape[-1]
        mu = posterior_mean_out
        m = prior_mean_out
        logE = posterior_log_var_out
        logS = prior_log_var_out
        KLD = 0.5 * (th.sum(logS, dim=-1) - th.sum(logE, dim=-1) - gauss_dim + th.sum(
            1 / th.exp(logS) * th.exp(logE), dim=-1) + ((m - mu) / th.exp(logS) * (m - mu)).sum(dim=-1))
        KLD_error = KLD.reshape(batch_size, batch.max_seq_length - 1, self.n_agents)#.mean(-1).unsqueeze(-1)

        vae_error = production_error + KLD_error

        return vae_error, production_error, KLD_error

    def parameters(self):
        return self.exp_vae.vae_params

    def named_parameters(self):
        return self.exp_vae.vae_named_parameters

    def named_children(self):
        return self.exp_vae._named_children

    def to_device(self):
        self.exp_vae.to(device=self.device)

    def save_models(self, path, model_name="{}/intrinsic_vae{}.th"):
        th.save(self.exp_vae.state_dict(), model_name.format(path, ''))

    def load_models(self, path, model_name="{}/intrinsic_vae{}.th"):
        self.exp_vae.load_state_dict(th.load(model_name.format(path, ''), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.exp_vae = agent_REGISTRY[self.args.EXP_AGENT](self.history_input_shape, self.args)

    def init_hidden(self, batch_size):
        self.prior_encoder_hidden = self.exp_vae.prior_init_hidden()
        self.posterior_encoder_hidden = self.exp_vae.posterior_init_hidden()
        self.prior_encoder_hidden = self.prior_encoder_hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.posterior_encoder_hidden = self.posterior_encoder_hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        # Observations minus the initial observation
        # obs_inputs = batch["obs"][:, t] - batch["obs"][:, 0]
        obs_inputs = batch["obs"][:, t]
        inputs.append(obs_inputs)
        if self.args.OBS_LAST_ACTION:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.OBS_AGENT_ID:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        # current observation + last action + agent id
        input_shape = self.scheme["obs"]["vshape"]
        if self.args.OBS_LAST_ACTION:
            input_shape += self.scheme["actions_onehot"]["vshape"][0]
        if self.args.OBS_AGENT_ID:
            input_shape += self.n_agents

        return input_shape

    def update_noise_epsilon(self, train_steps):
        pass
