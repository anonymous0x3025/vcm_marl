import copy
from components.episode_buffer import EpisodeBatch
from learners import QLearner
import torch as th
from torch.optim import RMSprop
from modules.vae.vae import Encoder, Decoder, VAEModel
from modules.mixers.vi_exp_qmix import VIQMixer
from utils.rl_utils import overrides


class VIExpQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.logger = logger
        self.n_agents = args.N_AGENTS
        self.device = th.device(self.args.DEVICE)

        # VAE
        self.vae_model = VAEModel(encoder=Encoder(args.STATE_SHAPE, args), decoder=Decoder(args.STATE_SHAPE, args=args), args=args).to(self.device)
        self.vae_hidden_state = None
        self.vae_params = list(self.vae_model.parameters())
        self.params += self.vae_params

        self.mixer = self.mixer = VIQMixer(args)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        # re-configure optimizer
        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.last_target_update_episode = 0

    @overrides(QLearner)
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] * self.args.REWARD_SCALE
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.DOUBLE_Q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # next_states
        next_states = batch["state"][:, 1:]

        # VAE
        batch_size = batch["state"].shape[0]
        self.vae_hidden_state = self.vae_model.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)
        mean_out, log_var_out, z_out, x_hat_out, vae_hidden_state = [], [], [], [], []
        for t in range(batch.max_seq_length):
            mean, log_var, z, x_hat, self.vae_hidden_state = self.vae_model(batch["state"][:, t], self.vae_hidden_state)
            mean_out.append(mean)
            log_var_out.append(log_var)
            z_out.append(z)
            x_hat_out.append(x_hat)
            vae_hidden_state.append(self.vae_hidden_state)
        # Concat over time
        mean_out = th.stack(mean_out, dim=1)
        log_var_out = th.stack(log_var_out, dim=1)
        z_out = th.stack(z_out, dim=1)
        x_hat_out = th.stack(x_hat_out, dim=1)
        vae_hidden_state = th.stack(vae_hidden_state, dim=1)

        production_loss = (x_hat_out[:, :-1] - next_states).pow(2).mean()

        if self.args.KL_TO_GAUSS_PRIOR:
            KLD = (- 0.5 * (1 + log_var_out - mean_out.pow(2) - log_var_out.exp()).sum(dim=-1))
            # average across batch
            KLD = KLD.sum(dim=-1).mean()
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
            # average across tasks
            KLD = KLD.sum(dim=-1).mean()

        vae_loss = production_loss + KLD

        # MIX
        chosen_action_qvals = self.mixer(chosen_action_qvals, z_out[:, :-1])
        target_tot_qvals = self.target_mixer(target_max_qvals, z_out[:, 1:])

        # intrinsic reward
        next_z_out = []
        # no use N-step intrinsic reward
        if self.args.N_STEP <= 1:
            for t in range(batch.max_seq_length - 1):
                _, _, next_z, _, _ = self.vae_model(x_hat_out[:, t], vae_hidden_state[:, t])
                next_z_out.append(next_z)
        # N-step intrinsic reward
        else:
            for t in range(batch.max_seq_length - 1):
                tau = max(t - self.args.N_STEP + 1, 0)
                next_x_hat = x_hat_out[:, tau]
                next_vae_hidden_state = vae_hidden_state[:, tau]
                for _ in range(tau, t + 1):
                    _, _, next_z, next_x_hat, next_vae_hidden_state = self.vae_model(next_x_hat, next_vae_hidden_state)
                next_z_out.append(next_z)
        # Concat over time
        next_z_out = th.stack(next_z_out, dim=1)

        production_target_tot_qvals = self.target_mixer(target_max_qvals, next_z_out)
        intrinsic_rewards = (
                production_target_tot_qvals.detach() - target_tot_qvals.detach()
        ).pow(2)
        intrinsic_rewards = th.clip(
            intrinsic_rewards, min=self.args.CLIP_MIN_INT_REWARD, max=self.args.CLIP_MAX_INT_REWARD
        )
        rewards += self.args.BETA * intrinsic_rewards

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.GAMMA * (1 - terminated) * target_tot_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimizer.zero_grad()
        loss += vae_loss
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(parameters=self.params, max_norm=self.args.GRAD_NORM_CLIP)
        self.optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("vi_exp_intrinsic_rewards", intrinsic_rewards.mean().item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("vae_loss", vae_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.N_AGENTS),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        super()._update_targets()

    def to_device(self):
        super().to_device()

    def save_models(self, path):
        super().save_models(path)
        th.save(self.vae_model.state_dict(), "{}/vae.th".format(path))

    def load_models(self, path):
        super().load_models(path)
        self.vae_model.load_state_dict(th.load("{}/vae.th".format(path), map_location=lambda storage, loc: storage))
