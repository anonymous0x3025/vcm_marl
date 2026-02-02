import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch as th
from torch.optim import RMSprop


class DMAQ_qattenLearner_ForIntrinsic:
    def __init__(self, mac, intrinsic_mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.intrinsic_mac = intrinsic_mac
        self.intrinsic_params = list(self.intrinsic_mac.parameters())

        self.int_last_target_update_episode = 0
        self.last_target_update_episode = 0

        self.mixer = None
        if args.MIXER is not None:
            if args.MIXER == "dmaq":
                self.mixer = DMAQer(args)
            elif args.MIXER == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

            self.intrinsic_mixer = copy.deepcopy(self.mixer)
            self.intrinsic_params += list(self.intrinsic_mixer.parameters())
            self.target_intrinsic_mixer = copy.deepcopy(self.mixer)

        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.intrinsic_optimizer = RMSprop(params=self.intrinsic_params, lr=args.INTRINSIC_LR, alpha=args.OPTIM_ALPHA,
                                           eps=args.OPTIM_EPS)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.target_intrinsic_mac = copy.deepcopy(intrinsic_mac)

        self.log_stats_t = -self.args.LEARNER_LOG_INTERVAL - 1

        self.n_actions = self.args.N_ACTIONS

        self.device = th.device(self.args.DEVICE)

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, intrinsic_mac, mixer, intrinsic_mixer, optimizer, intrinsic_optimizer, params, intrinsic_params, vae_error=None, production_error=None, KLD_error=None, z_out=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] * self.args.REWARD_SCALE
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        mac_out = []
        intrinsic_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = intrinsic_mac.intrinsic_forward(z_out[:, t], batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_intrinsic_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_intrinsic_mac.intrinsic_forward(z_out[:, t], batch, t=t)
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
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,))
            cur_max_actions_onehot = cur_max_actions_onehot.to(device=self.device)
            cur_max_actions = cur_max_actions.to(device=self.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_intrinsic_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_intrinsic_mac.intrinsic_forward(z_out[:, t], batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if intrinsic_mixer is not None:
            if self.args.MIXER == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    intrinsic_mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = intrinsic_mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = intrinsic_mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = intrinsic_mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.DOUBLE_Q:
                if self.args.MIXER == "dmaq_qatten":
                    target_chosen, _, _ = self.target_intrinsic_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_intrinsic_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_intrinsic_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_intrinsic_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_intrinsic_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals.detach()

        # Td-error
        td_error = (chosen_action_qvals - targets)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.MIXER == "dmaq_qatten":
            loss_for_int_q = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss_for_int_q = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        int_hit_prob = masked_hit_prob.sum() / mask.sum()

        production_error = production_error
        production_error_detached = production_error.clone().detach().mean(-1).unsqueeze(-1)

        # Optimize
        vae_loss = []
        for i in range(self.args.N_AGENTS):
            vae_loss.append(vae_error[:, :, i].mean())
        vae_loss = th.stack(vae_loss, dim=0)

        total_intrinsic_loss = loss_for_int_q + vae_loss.sum(dim=0)

        intrinsic_optimizer.zero_grad()
        total_intrinsic_loss.backward()
        int_grad_norm = th.nn.utils.clip_grad_norm_(parameters=intrinsic_params, max_norm=self.args.INTRINSIC_GRAD_NORM_CLIP)
        intrinsic_optimizer.step()

        # Calculate estimated Q-Values
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.intrinsic_forward(z_out[:, t].clone().detach(), batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.intrinsic_forward(z_out[:, t].clone().detach(), batch, t=t)
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
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,))
            cur_max_actions_onehot = cur_max_actions_onehot.to(device=self.device)
            cur_max_actions = cur_max_actions.to(device=self.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.intrinsic_forward(z_out[:, t].clone().detach(), batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            if self.args.MIXER == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.DOUBLE_Q:
                if self.args.MIXER == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # intrinsic rewards
        intrinsic_rewards = self.args.BETA * th.clip(
            production_error_detached, min=0.0, max=self.args.CLIP_INTRINSIC_REWARD
        )

        # Calculate 1-step Q-Learning targets
        targets = rewards + intrinsic_rewards.detach() + self.args.GAMMA * (1 - terminated) * target_max_qvals.detach()

        # Td-error
        td_error = (chosen_action_qvals - targets)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.MIXER == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.GRAD_NORM_CLIP)
        optimizer.step()

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("intrinsic_rewards", intrinsic_rewards.mean().item(), t_env)
            self.logger.log_stat("total_intrinsic_loss", total_intrinsic_loss.item(), t_env)
            self.logger.log_stat("loss_for_int_q", loss_for_int_q.item(), t_env)
            self.logger.log_stat("int_hit_prob", int_hit_prob.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("int_grad_norm", int_grad_norm, t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.N_AGENTS),
                                 t_env)
            self.log_stats_t = t_env

    def _train(self, batch: EpisodeBatch, t_env: int, episode_num: int, vae_error=None, production_error=None, KLD_error=None, z_out=None):
        self.sub_train(batch, t_env, episode_num, self.mac, self.intrinsic_mac, self.mixer, self.intrinsic_mixer, self.optimizer, self.intrinsic_optimizer, self.params, self.intrinsic_params, vae_error, production_error, KLD_error, z_out)
        if (episode_num - self.int_last_target_update_episode) / self.args.INTRINSIC_TARGET_UPDATE_INTERVAL >= 1.0:
            self._int_update_targets()
            self.int_last_target_update_episode = episode_num
        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self._train(batch, t_env, episode_num)

    def _int_update_targets(self):
        self.target_intrinsic_mac.load_state(self.intrinsic_mac)
        if self.intrinsic_mixer is not None:
            self.target_intrinsic_mixer.load_state_dict(self.intrinsic_mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        self.mac.to_device()
        self.target_mac.to_device()
        self.intrinsic_mac.to_device()
        self.target_intrinsic_mac.to_device()
        if self.mixer is not None:
            self.mixer.to(device=self.mac.device)
            self.target_mixer.to(device=self.mac.device)
            self.intrinsic_mixer.to(device=self.intrinsic_mac.device)
            self.target_intrinsic_mixer.to(device=self.intrinsic_mac.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))