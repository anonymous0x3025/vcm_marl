import copy
from components.episode_buffer import EpisodeBatch
from learners.q_learner import QLearner
import torch as th
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from torch.optim import RMSprop


class SurpriseQLearner(QLearner):
    def __init__(self, mac, exp_mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.logger = logger
        self.n_agents = args.N_AGENTS

        # curiosity
        self.exp_mac = exp_mac
        self.exp_params = self.exp_mac.parameters()

        # intrinsic mac
        if args.USE_INTRINSIC_MAC:
            self.int_last_target_update_episode = 0

            self.intrinsic_mac = copy.deepcopy(mac)
            self.exp_params += list(self.intrinsic_mac.parameters())

            self.intrinsic_mixer = None
            if args.MIXER is not None:
                if args.MIXER == "vdn":
                    self.intrinsic_mixer = VDNMixer()
                elif args.MIXER == "qmix":
                    self.intrinsic_mixer = QMixer(args)
                else:
                    raise ValueError("Mixer {} not recognised.".format(args.MIXER))
                self.exp_params += list(self.intrinsic_mixer.parameters())
                self.target_intrinsic_mixer = copy.deepcopy(self.intrinsic_mixer)

            # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
            self.target_intrinsic_mac = copy.deepcopy(self.intrinsic_mac)

        # re-configure optimizer
        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)
        self.exp_optimizer = RMSprop(params=self.exp_params, lr=args.EXP_LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # VAE
        self.exp_mac.init_hidden(batch.batch_size)
        prior_mean_out, prior_log_var_out, prior_z_out = [], [], []
        posterior_mean_out, posterior_log_var_out, posterior_z_out = [], [], []
        for t in range(batch.max_seq_length):
            mean, log_var, z = self.exp_mac.prior_encoder_forward(batch, t=t)
            prior_mean_out.append(mean)
            prior_log_var_out.append(log_var)
            prior_z_out.append(z)

            _mean, _log_var, _z = self.exp_mac.posterior_encoder_forward(batch, t=t)
            posterior_mean_out.append(_mean)
            posterior_log_var_out.append(_log_var)
            posterior_z_out.append(_z)

        # Concat over time
        prior_mean_out = th.stack(prior_mean_out, dim=1)
        prior_log_var_out = th.stack(prior_log_var_out, dim=1)
        prior_z_out = th.stack(prior_z_out, dim=1)
        posterior_mean_out = th.stack(posterior_mean_out, dim=1)
        posterior_log_var_out = th.stack(posterior_log_var_out, dim=1)
        posterior_z_out = th.stack(posterior_z_out, dim=1)

        vae_error, production_error, KLD_error = self.exp_mac.decoder_forward(
            batch,
            prior_mean_out[:, :-1], posterior_mean_out[:, :-1],
            prior_log_var_out[:, :-1], posterior_log_var_out[:, :-1],
            posterior_z_out[:, :-1]
        )

        if self.args.INDIVIDUAL_UPDATE:
            intrinsic_rewards = self.args.BETA * th.clip(
                KLD_error.clone().detach(),
                min=0,
                max=self.args.MAX_INTRINSIC_REWARD
            )
        else:
            intrinsic_rewards = self.args.BETA * th.clip(
                KLD_error.mean(dim=2).unsqueeze(2).clone().detach(),
                min=0,
                max=self.args.MAX_INTRINSIC_REWARD
            )

        vae_loss = vae_error.mean()

        if self.args.USE_INTRINSIC_MAC:
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1] * self.args.REWARD_SCALE
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            # Calculate estimated Q-Values
            mac_out = []
            self.intrinsic_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.intrinsic_mac.intrinsic_forward(prior_z_out[:, t], batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_intrinsic_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_intrinsic_mac.intrinsic_forward(prior_z_out[:, t], batch, t=t)
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

            if self.intrinsic_mixer is not None:
                chosen_action_qvals = self.intrinsic_mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_intrinsic_mixer(target_max_qvals, batch["state"][:, 1:])

            # Calculate 1-step Q-Learning targets
            targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals.detach()

            # Td-error
            td_error = (chosen_action_qvals - targets)

            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum()
            exp_loss = loss + vae_loss

            # Optimize
            self.exp_optimizer.zero_grad()
            exp_loss.backward()
            exp_grad_norm = th.nn.utils.clip_grad_norm_(parameters=self.exp_params, max_norm=self.args.GRAD_NORM_CLIP)

            if (episode_num - self.int_last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
                self._update_intrinsic_targets()
                self.int_last_target_update_episode = episode_num

            if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
                self.logger.log_stat("exp_grad_norm", exp_grad_norm, t_env)

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("VAE loss", vae_loss.item(), t_env)
            self.logger.log_stat("production loss", production_error.mean().item(), t_env)
            self.logger.log_stat("KLD loss", KLD_error.mean().item(), t_env)
            self.logger.log_stat("intrinsic_rewards", intrinsic_rewards.mean().item(), t_env)

        # Train MARL Algorithm
        if self.args.USE_INTRINSIC_MAC:
            self._train(batch, t_env, episode_num, additional_loss=None, intrinsic_rewards=intrinsic_rewards, z_out=prior_z_out)
        else:
            self.exp_optimizer.zero_grad()
            self._train(batch, t_env, episode_num, additional_loss=vae_loss, intrinsic_rewards=intrinsic_rewards, z_out=prior_z_out)
        self.exp_optimizer.step()

    def _update_intrinsic_targets(self):
        self.target_intrinsic_mac.load_state(self.intrinsic_mac)
        if self.intrinsic_mixer is not None:
            self.target_intrinsic_mixer.load_state_dict(self.intrinsic_mixer.state_dict())

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()
        if self.args.USE_INTRINSIC_MAC:
            self.intrinsic_mac.to_device()
            self.target_intrinsic_mac.to_device()
            if self.intrinsic_mixer is not None:
                self.intrinsic_mixer.to(device=self.intrinsic_mac.device)
                self.target_intrinsic_mixer.to(device=self.intrinsic_mac.device)

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)
        self.intrinsic_mac.save_models(path, model_name="{}/intrinsic_agent.th")

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
        self.intrinsic_mac.load_models(path, model_name="{}/intrinsic_agent.th")
