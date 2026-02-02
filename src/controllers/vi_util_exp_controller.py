from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
import torch.nn.functional as F


# This multi-agent controller shares parameters between agents
class VIUtilExpMAC:
    def __init__(self, scheme, args):
        self.n_agents = args.N_AGENTS
        self.args = args
        self.scheme = scheme
        self.encoder_input_shape = self._get_input_shape()
        self.decoder_input_shape = args.OBS_ENCODING_DIM + args.ACTION_ENCODING_DIM + args.LATENT_DIM
        self.obs_dim = self.scheme["obs"]["vshape"]
        self.action_dim = args.N_ACTIONS
        self.decoder_output_shape = self.obs_dim
        self._build_agents()
        self.agent_output_type = args.AGENT_OUTPUT_TYPE

        self.vae_hidden = None
        self.util_hidden = None

        self.device = th.device(self.args.DEVICE)

    def encoder_forward(self, ep_batch, t):
        # VAE forwarding to make intrinsic reward
        # batch_size = ep_batch.batch_size
        # obs = ep_batch["obs"][:, t]
        # obs = obs.view(batch_size, self.n_agents, -1)
        vae_inputs = self._build_inputs(ep_batch, t)
        mean, log_var, z, self.vae_hidden = self.exp_vae.encoder_forward(vae_inputs, self.vae_hidden)
        return mean, log_var, z

    def util_forward(self, batch_size, z_out, target=False):
        util_q, self.util_hidden = self.exp_vae.util_forward(z_out, self.util_hidden, target)
        return util_q.view(batch_size, self.n_agents, -1)

    def calc_vae_loss(self, batch, mean_out, log_var_out, z_out):
        batch_size = batch.batch_size

        # next_obs = batch["obs"][:, 1:]
        # y_next_obs = []
        # decoder_inputs = []
        # for t in range(batch.max_seq_length - 1):
        #     _next_obs = next_obs[:, t].view(batch_size, self.n_agents, -1)
        #     y_next_obs.append(self._build_inputs(batch, _next_obs))
        #     _z_out = z_out[:, t].view(batch_size, self.n_agents, -1)
        #     decoder_inputs.append(self._build_decoder_inputs(batch, _z_out, t))
        # y_next_obs = th.stack(y_next_obs, dim=1)
        # y_next_obs = y_next_obs.reshape(batch_size, batch.max_seq_length - 1, self.n_agents, -1)

        # decoder_inputs = []
        # for t in range(batch.max_seq_length - 1):
        #     _z_out = z_out[:, t].view(batch_size, self.n_agents, -1)
        #     decoder_inputs.append(self._build_decoder_inputs(batch, _z_out, t))

        obs = batch["obs"][:, :-1]
        obs = obs.reshape(batch_size * self.n_agents, batch.max_seq_length - 1, -1)
        actions_onehot = batch["actions_onehot"][:, :-1]
        actions_onehot = actions_onehot.reshape(batch_size*self.n_agents, batch.max_seq_length - 1, -1)

        # Observations minus the initial observation
        # next_obs = th.sub(batch["obs"][:, 1:], batch["obs"][:, 0].unsqueeze(1))
        next_obs = batch["obs"][:, 1:]
        y_next_obs = next_obs.reshape(batch_size, batch.max_seq_length - 1, self.n_agents, -1)

        # decoder_inputs = th.stack(decoder_inputs, dim=1)
        x_hat_out = self.exp_vae.decoder_forward(z_out, obs, actions_onehot)
        x_hat_out = x_hat_out.reshape(batch_size, batch.max_seq_length - 1, self.n_agents, -1)

        # reconstruction loss
        production_loss = (x_hat_out - y_next_obs).pow(2).sum(-1)

        if self.args.KL_TO_GAUSS_PRIOR:
            KLD = (- 0.5 * (1 + log_var_out - mean_out.pow(2) - log_var_out.exp()).sum(dim=-1))
        else:
            gauss_dim = mean_out.shape[-1]
            # add the gaussian prior
            all_means = th.cat((th.zeros(1, *mean_out.shape[1:]).to(self.device), mean_out))
            all_logvars = th.cat((th.zeros(1, *log_var_out.shape[1:]).to(self.device), log_var_out))
            
            
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            KLD = 0.5 * (th.sum(logS, dim=-1) - th.sum(logE, dim=-1) - gauss_dim + th.sum(
                1 / th.exp(logS) * th.exp(logE), dim=-1) + ((m - mu) / th.exp(logS) * (m - mu)).sum(dim=-1))
        KLD = KLD.reshape(batch_size, batch.max_seq_length - 1, self.n_agents)
        vae_loss = self.args.BETA * (production_loss.mean() + KLD.mean())
        return production_loss.detach(), KLD.detach(), vae_loss

    def update_noise_epsilon(self, steps=0, train_steps=0):
        pass

    def parameters(self):
        return self.exp_vae.vae_params, self.exp_vae.util_params, self.exp_vae.target_util_params

    def load_state(self):
        self.exp_vae.load_state()

    def to_device(self):
        self.exp_vae.to(device=self.device)

    def save_models(self, path, model_name="{}/exp_vae{}.th"):
        th.save(self.exp_vae.state_dict(), model_name.format(path, ''))

    def load_models(self, path, model_name="{}/exp_vae{}.th"):
        self.exp_vae.load_state_dict(th.load(model_name.format(path, ''), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.exp_vae = agent_REGISTRY[self.args.EXP_AGENT](self.encoder_input_shape, self.obs_dim, self.action_dim, self.decoder_input_shape, self.decoder_output_shape, self.args)
        self.exp_vae.share_memory()

    def init_hidden(self, batch_size):
        self.vae_hidden, self.util_hidden = self.exp_vae.init_hidden()
        self.vae_hidden = self.vae_hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.util_hidden = self.util_hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

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
        input_shape = self.scheme["obs"]["vshape"]
        if self.args.OBS_LAST_ACTION:
            input_shape += self.scheme["actions_onehot"]["vshape"][0]
        if self.args.OBS_AGENT_ID:
            input_shape += self.n_agents

        return input_shape
