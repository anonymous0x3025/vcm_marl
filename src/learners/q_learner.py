import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

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

        self.optimizer = RMSprop(params=self.params, lr=args.LR, alpha=args.OPTIM_ALPHA, eps=args.OPTIM_EPS)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.LEARNER_LOG_INTERVAL - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self._train(batch, t_env, episode_num)

    def _train(self, batch, t_env, episode_num, additional_loss=None, intrinsic_rewards=None, z_out=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] * self.args.REWARD_SCALE
        if intrinsic_rewards is not None:
            rewards = rewards + intrinsic_rewards
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if z_out is None:
                agent_outs = self.mac.forward(batch, t=t)
            else:
                agent_outs = self.mac.intrinsic_forward(z_out[:, t].detach(), batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if z_out is None:
                target_agent_outs = self.target_mac.forward(batch, t=t)
            else:
                target_agent_outs = self.target_mac.intrinsic_forward(z_out[:, t].detach(), batch, t=t)
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
            if "INDIVIDUAL_UPDATE" in dir(self.args) and self.args.INDIVIDUAL_UPDATE and intrinsic_rewards is not None:
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

                input_state = batch["state"][:, :-1].unsqueeze(0).expand(self.args.N_AGENTS, -1, -1, -1)
                input_state = input_state.reshape(-1, input_state.shape[2], input_state.shape[3])

                individual_chosen_action_qvals = self.mixer(individual_chosen_action_qvals, input_state)
                individual_chosen_action_qvals = individual_chosen_action_qvals.reshape(self.args.N_AGENTS, batch.batch_size, -1, 1)
                individual_chosen_action_qvals = individual_chosen_action_qvals.permute(1, 2, 0, 3).squeeze(3)
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.GAMMA * (1 - terminated) * target_max_qvals.detach()

        # Td-error
        if "INDIVIDUAL_UPDATE" in dir(self.args) and self.args.INDIVIDUAL_UPDATE:
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
        if additional_loss is not None:
            loss += additional_loss
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(parameters=self.params, max_norm=self.args.GRAD_NORM_CLIP)
        self.optimizer.step()

        if (episode_num - self.last_target_update_episode) / self.args.TARGET_UPDATE_INTERVAL >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.LEARNER_LOG_INTERVAL:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.N_AGENTS), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.N_AGENTS), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.log_info("Updated target network")

    def to_device(self):
        self.mac.to_device()
        self.target_mac.to_device()
        if self.mixer is not None:
            self.mixer.to(device=self.mac.device)
            self.target_mixer.to(device=self.mac.device)

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
