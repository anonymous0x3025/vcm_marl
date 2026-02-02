import copy
from components.episode_buffer import EpisodeBatch
from learners.dmaq_qatten_learner import DMAQ_qattenLearner
import torch as th
from torch.optim import RMSprop
from utils.rl_utils import overrides
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.attention import SelfAttention
from components.epsilon_schedules import DecayThenFlatSchedule


class VIIntrinsicQplexLearner(DMAQ_qattenLearner):
    def __init__(self, mac, exp_mac, scheme, logger, args, intrinsic_mac=None):
        super().__init__(mac, scheme, logger, args)
        self.logger = logger
        self.n_agents = args.N_AGENTS

        # strangeness index module
        self.exp_mac = exp_mac
        self.exp_params = self.exp_mac.parameters()

        # intrinsic mac
        if args.USE_INTRINSIC_MAC:
            self.int_last_target_update_episode = 0

            self.intrinsic_mac = intrinsic_mac
            self.exp_params += list(self.intrinsic_mac.parameters())

            self.intrinsic_mixer = None
            if args.MIXER is not None:
                self.intrinsic_mixer = QMixer(args)
                self.exp_params += list(self.intrinsic_mixer.parameters())
                self.target_intrinsic_mixer = copy.deepcopy(self.intrinsic_mixer)

            # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
            self.target_intrinsic_mac = copy.deepcopy(self.intrinsic_mac)

        if args.USE_ATTENTION:
            self.attention = SelfAttention(args)
            self.exp_params += list(self.attention.parameters())

        # re-configure optimizer
        self.exp_optimizer = RMSprop(params=self.exp_params, lr=args.INTRINSIC_LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.device = th.device(self.args.DEVICE)

        self.schedule = DecayThenFlatSchedule(
            args.INT_REWARD_SCALE_MAX, args.INT_REWARD_SCALE_MIN, args.INT_REWARD_SCALE_ANNEAL_TIME, decay="linear"
        )
        self.int_reward_scale = self.schedule.eval(0)

    def update_int_reward_scale(self, train_steps):
        self.int_reward_scale = self.schedule.eval(train_steps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # VAE
        self.exp_mac.init_hidden(batch.batch_size)
        mean_out, log_var_out, z_out, z_for_policy_out = [], [], [], []
        for t in range(batch.max_seq_length):
            mean, log_var, z, z_for_policy = self.exp_mac.encoder_forward(batch, t=t)
            mean_out.append(mean)
            log_var_out.append(log_var)
            z_out.append(z)
            z_for_policy_out.append(z_for_policy)
        # Concat over time
        mean_out = th.stack(mean_out, dim=1)
        log_var_out = th.stack(log_var_out, dim=1)
        z_out = th.stack(z_out, dim=1)
        z_for_policy_out = th.stack(z_for_policy_out, dim=1)

        if self.args.USE_ATTENTION:
            z_out_att = self.attention(z_out)

        if self.args.USE_ATTENTION:
            vae_error, production_error, KLD_error = self.exp_mac.decoder_forward(
                batch, mean_out[:, :-1], log_var_out[:, :-1], z_out_att[:, :-1]
            )
        else:
            vae_error, production_error, KLD_error = self.exp_mac.decoder_forward(
                batch, mean_out[:, :-1], log_var_out[:, :-1], z_out[:, :-1]
            )

        # Optimize
        vae_loss = vae_error.mean()

        if self.args.USE_INTRINSIC_MAC:
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1] * self.args.REWARD_SCALE
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            mac_out = []
            self.intrinsic_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                if self.args.USE_ATTENTION:
                    agent_outs = self.intrinsic_mac.intrinsic_forward(z_out_att[:, t], batch, t=t)
                else:
                    agent_outs = self.intrinsic_mac.intrinsic_forward(z_out[:, t], batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            vals = mac_out[:, :-1].squeeze(3)
            # vals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            target_mac_out = []
            self.target_intrinsic_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                if self.args.USE_ATTENTION:
                    target_agent_outs = self.target_intrinsic_mac.intrinsic_forward(z_out_att[:, t], batch, t=t)
                else:
                    target_agent_outs = self.target_intrinsic_mac.intrinsic_forward(z_out[:, t], batch, t=t)
                target_mac_out.append(target_agent_outs)

            target_vals = th.stack(target_mac_out[1:], dim=1).squeeze(3)  # Concat across time
            # # Mask out unavailable actions
            # target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            #
            # target_vals = target_mac_out.max(dim=3)[0]

            if self.intrinsic_mixer is not None:
                vals = self.intrinsic_mixer(vals, batch["state"][:, :-1])
                target_vals = self.target_intrinsic_mixer(target_vals, batch["state"][:, 1:])

            # Calculate 1-step Q-Learning targets
            targets_for_int_v = rewards + self.args.GAMMA * (1 - terminated) * target_vals

            # Td-error
            td_error = vals - targets_for_int_v.detach()
            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask
            # masked_td_error_detached = (masked_td_error.clone().detach() ** 2)

            # Normal L2 loss, take mean over actual data
            loss_for_int_v = (masked_td_error ** 2).sum() / mask.sum()

            # vae_error_detached = vae_error.clone().detach().mean(-1).unsqueeze(-1)
            # KLD_error_detached = KLD_error.clone().detach().mean(-1).unsqueeze(-1)

            total_intrinsic_loss = loss_for_int_v + vae_loss
        else:
            total_intrinsic_loss = vae_loss

        self.exp_optimizer.zero_grad()
        total_intrinsic_loss.backward()
        exp_grad_norm = th.nn.utils.clip_grad_norm_(parameters=self.exp_params, max_norm=self.args.GRAD_NORM_CLIP)

        if self.args.USE_INTRINSIC_MAC:
            if (episode_num - self.int_last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
                self._update_intrinsic_targets()
                self.int_last_target_update_episode = episode_num

            if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
                self.logger.log_stat("exp_grad_norm", exp_grad_norm, t_env)
                self.logger.log_stat("total_intrinsic_loss", total_intrinsic_loss.item(), t_env)
                self.logger.log_stat("loss_for_int_v", loss_for_int_v.item(), t_env)
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("int_v_taken_mean",
                                     (vals * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
                self.logger.log_stat("int_target_mean",
                                     (targets_for_int_v * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
                self.logger.log_stat("target_mean", (targets_for_int_v * mask).sum().item() / (mask_elems * self.args.N_AGENTS),
                                     t_env)

        production_error_detached = production_error.clone().detach().mean(-1).unsqueeze(-1)
        intrinsic_rewards = self.int_reward_scale * self.args.BETA * th.clip(
            production_error_detached, min=0.0, max=self.args.CLIP_INTRINSIC_REWARD
        )

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("intrinsic reward scale", self.int_reward_scale, t_env)
            self.logger.log_stat("VAE loss", vae_loss.item(), t_env)
            self.logger.log_stat("production loss", production_error.mean().item(), t_env)
            self.logger.log_stat("KLD loss", KLD_error.mean().item(), t_env)
            self.logger.log_stat("intrinsic_rewards", intrinsic_rewards.mean().item(), t_env)
            self.log_stats_t = t_env

        # Train MARL Algorithm
        if self.args.USE_INT_REWARD:
            self._train(batch, t_env, episode_num, intrinsic_rewards=intrinsic_rewards, z_out=z_for_policy_out)
        else:
            self._train(batch, t_env, episode_num, intrinsic_rewards=None, z_out=z_for_policy_out)
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
        if self.args.USE_ATTENTION:
            self.attention.to(device=self.mac.device)

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)
        self.intrinsic_mac.save_models(path, model_name="{}/intrinsic_agent.th")

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
        self.intrinsic_mac.load_models(path, model_name="{}/intrinsic_agent.th")