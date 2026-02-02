import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop


class QLearnerForIntrinsic:
    def __init__(self, mac, intrinsic_mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.exp_mac = None
        self.logger = logger

        self.params = list(mac.parameters())

        self.intrinsic_mac = intrinsic_mac
        self.intrinsic_params = list(self.intrinsic_mac.parameters())

        self.int_last_target_update_episode = 0
        self.last_target_update_episode = 0

        self.mixer = None
        if args.MIXER is not None:
            if args.MIXER == "vdn":
                self.mixer = VDNMixer()
            elif args.MIXER == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.MIXER))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

            self.intrinsic_mixer = copy.deepcopy(self.mixer)
            self.intrinsic_params += list(self.intrinsic_mixer.parameters())
            self.target_intrinsic_mixer = copy.deepcopy(self.mixer)

        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        self.intrinsic_optimizer = RMSprop(params=self.intrinsic_params, lr=args.INTRINSIC_LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_intrinsic_mac = copy.deepcopy(intrinsic_mac)

        self.log_stats_t = -self.args.LEARNER_LOG_INTERVAL - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self._train(batch, t_env, episode_num)

    def _train(self, batch, t_env, episode_num, vae_error=None, production_error=None, KLD_error=None, z_out=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] * self.args.REWARD_SCALE
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Intrinsic update
        # Calculate estimated Q-Values
        mac_out = []
        self.intrinsic_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.intrinsic_mac.intrinsic_forward(z_out[:, t], batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

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
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.intrinsic_mixer is not None:
            chosen_action_qvals = self.intrinsic_mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_intrinsic_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets_for_int_q = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = chosen_action_qvals - targets_for_int_q.detach()
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # masked_td_error_detached = (masked_td_error.clone().detach() ** 2)

        # Normal L2 loss, take mean over actual data
        loss_for_int_q = (masked_td_error ** 2).sum() / mask.sum()

        # vae_error_detached = vae_error.clone().detach().mean(-1).unsqueeze(-1)
        # KLD_error_detached = KLD_error.clone().detach().mean(-1).unsqueeze(-1)

        # Optimize
        vae_loss = []
        for i in range(self.args.N_AGENTS):
            vae_loss.append(vae_error[:, :, i].mean())
        vae_loss = th.stack(vae_loss, dim=0)

        total_intrinsic_loss = loss_for_int_q + vae_loss.sum(dim=0)

        self.intrinsic_optimizer.zero_grad()
        total_intrinsic_loss.backward()
        int_grad_norm = th.nn.utils.clip_grad_norm_(parameters=self.intrinsic_params, max_norm=self.args.INTRINSIC_GRAD_NORM_CLIP)
        self.intrinsic_optimizer.step()

        # Q update
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.intrinsic_forward(z_out[:, t].clone().detach(), batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

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
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            if self.args.INDIVIDUAL_UPDATE:
                individual_chosen_action_qvals = []
                for agent_id in range(self.args.N_AGENTS):
                    other_ids = [i for i in range(self.args.N_AGENTS) if not i == agent_id]
                    each_chosen_action_qvals = th.cat(
                        [chosen_action_qvals[:, :, agent_id].unsqueeze(2),
                         chosen_action_qvals[:, :, other_ids].clone().detach()],
                        dim=2
                    )
                    individual_chosen_action_qvals.append(each_chosen_action_qvals)
                individual_chosen_action_qvals = th.stack(individual_chosen_action_qvals, dim=0)
                individual_chosen_action_qvals = individual_chosen_action_qvals.reshape(-1, individual_chosen_action_qvals.shape[2], individual_chosen_action_qvals.shape[3])
                # print(individual_chosen_action_qvals.shape)

                input_state = batch["state"][:, :-1].unsqueeze(0).expand(self.args.N_AGENTS, -1, -1, -1)
                input_state = input_state.reshape(-1, input_state.shape[2], input_state.shape[3])

                individual_chosen_action_qvals = self.mixer(individual_chosen_action_qvals, input_state)
                individual_chosen_action_qvals = individual_chosen_action_qvals.reshape(self.args.N_AGENTS, batch.batch_size, -1, 1)
                individual_chosen_action_qvals = individual_chosen_action_qvals.permute(1, 2, 0, 3).squeeze(3)
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # KLD 혹은 intrinsic Q loss를 사용하면 안됨, 이유는 실제 Q loss가 높여야할 기대보상값과 정반대되기 때문
        # 즉, KLD 혹은 intrinsic Q loss는 에이전트에게 도움이 되는 latent를 만드는 것일뿐임
        # intrinsic_rewards = self.args.BETA*production_error_detached
        if self.args.INDIVIDUAL_UPDATE:
            production_error_detached = production_error.clone().detach()
            intrinsic_rewards = self.args.BETA*th.clip(
                production_error_detached, min=0.0, max=self.args.CLIP_INTRINSIC_REWARD
            )
        else:
            production_error_detached = production_error.clone().detach().mean(-1).unsqueeze(-1)
            intrinsic_rewards = self.args.BETA*th.clip(
                production_error_detached, min=0.0, max=self.args.CLIP_INTRINSIC_REWARD
            )
        # intrinsic_rewards = self.args.BETA*KLD_error_detached
        # intrinsic_rewards = self.args.BETA * (masked_td_error_detached + vae_error_detached)
        # if episode_num > self.args.INITIATION_GIVING_INT_REWARD:
        targets = rewards + intrinsic_rewards.detach() + self.args.GAMMA * (1 - terminated) * target_max_qvals.detach()
        # else:
        #     targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals

        # Td-error
        if self.args.INDIVIDUAL_UPDATE:
            td_error = individual_chosen_action_qvals - targets.detach()
        else:
            td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(parameters=self.params, max_norm=self.args.GRAD_NORM_CLIP)
        self.optimizer.step()

        if (episode_num - self.int_last_target_update_episode) / self.args.INTRINSIC_TARGET_UPDATE_INTERVAL >= 1.0:
            self._int_update_targets()
            self.int_last_target_update_episode = episode_num

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("intrinsic_rewards", intrinsic_rewards.mean().item(), t_env)
            self.logger.log_stat("total_intrinsic_loss", total_intrinsic_loss.item(), t_env)
            self.logger.log_stat("loss_for_int_q", loss_for_int_q.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("int_grad_norm", int_grad_norm, t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.N_AGENTS), t_env)
            self.logger.log_stat("int_target_mean", (targets_for_int_q * mask).sum().item()/(mask_elems * self.args.N_AGENTS), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.N_AGENTS), t_env)
            self.log_stats_t = t_env

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
        self.optimizer.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


def model_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        model_freeze(child)


def model_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        model_unfreeze(child)
