from components.episode_buffer import EpisodeBatch
from learners import QLearner
import torch as th
from torch.optim import RMSprop
from utils.rl_utils import overrides
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


class VIUtilExpQLearner(QLearner):
    def __init__(self, mac, exp_mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.logger = logger
        self.n_agents = args.N_AGENTS

        # strangeness index module
        self.exp_mac = exp_mac
        self.vae_params, self.util_params, self.target_util_params = self.exp_mac.parameters()

        # re-configure optimizer
        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)
        self.vae_optimizer = RMSprop(params=self.vae_params, lr=args.UTIL_LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)
        self.util_optimizer = RMSprop(params=self.util_params, lr=args.UTIL_LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.last_target_update_episode = 0

        self.device = th.device(self.args.DEVICE)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # VAE
        self.exp_mac.init_hidden(batch.batch_size)
        mean_out, log_var_out, z_out, util_q_out = [], [], [], []
        for t in range(batch.max_seq_length):
            mean, log_var, z = self.exp_mac.encoder_forward(batch, t=t)
            util_q = self.exp_mac.util_forward(batch.batch_size, z.clone().detach())
            mean_out.append(mean)
            log_var_out.append(log_var)
            z_out.append(z)
            util_q_out.append(util_q)
        # Concat over time
        mean_out = th.stack(mean_out, dim=1)
        log_var_out = th.stack(log_var_out, dim=1)
        z_out = th.stack(z_out, dim=1)
        util_q_out = th.stack(util_q_out, dim=1)

        # Pick the util Q-Values for the actions taken by each agent
        chosen_util_q_vals = th.gather(util_q_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_util_q_out = []
        self.exp_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            z_out_detach = z_out[:, t].detach()
            target_util_q = self.exp_mac.util_forward(batch.batch_size, z_out_detach, target=True)
            target_util_q_out.append(target_util_q)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_util_q_out = th.stack(target_util_q_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_util_q_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.DOUBLE_Q:
            # Get actions that maximise live Q (for double q-learning)
            util_q_out_detach = util_q_out.clone().detach()
            util_q_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = util_q_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_util_max_qvals = th.gather(target_util_q_out, 3, cur_max_actions).squeeze(3)
        else:
            target_util_max_qvals = target_util_q_out.max(dim=3)[0]

        mean_out = mean_out[:, :-1]
        log_var_out = log_var_out[:, :-1]
        production_loss, KLD, vae_loss = self.exp_mac.calc_vae_loss(batch, mean_out, log_var_out, z_out[:, :-1])

        self.logger.log_stat("production_loss", production_loss.mean().item(), t_env)
        self.logger.log_stat("KLD", KLD.mean().item(), t_env)
        self.logger.log_stat("vae_loss", vae_loss.item(), t_env)

        int_rewards = self.args.BETA * (production_loss + KLD)
        self.logger.log_stat("int_rewards", int_rewards.mean().item(), t_env)

        # Calculate 1-step Q-Learning targets
        targets = int_rewards + self.args.GAMMA * (1 - terminated) * target_util_max_qvals

        # Td-error
        td_error = (chosen_util_q_vals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        util_q_loss = (masked_td_error ** 2).sum() / mask.sum()

        self.util_optimizer.zero_grad()
        util_q_loss.backward()
        th.nn.utils.clip_grad_norm_(parameters=self.util_params, max_norm=self.args.GRAD_NORM_CLIP)
        self.util_optimizer.step()

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        th.nn.utils.clip_grad_norm_(parameters=self.vae_params, max_norm=self.args.GRAD_NORM_CLIP)
        self.vae_optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.EXP_TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_exp_targets()

        # Train MARL Algorithm
        self._train(batch, t_env, episode_num, z_out=z_out)

    def _update_exp_targets(self):
        self.exp_mac.load_state()
        # self.logger.log_info("Updated exp target network")

    def to_device(self):
        super().to_device()
        self.exp_mac.to_device()

    def save_models(self, path):
        super().save_models(path)
        self.exp_mac.save_models(path)

    def load_models(self, path):
        super().load_models(path)
        self.exp_mac.load_models(path)
