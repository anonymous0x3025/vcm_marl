import ray

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import threading
import numpy as np
from modules.bandits.uniform import Uniform
from modules.bandits.reinforce_hierarchial import EZ_agent as enza
from modules.bandits.returns_bandit import ReturnsBandit as RBandit
from utils.rl_utils import convert_format_to_numpy_array


class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.BATCH_SIZE_RUN

        env_fn = env_REGISTRY[self.args.ENV]

        self.env_workers = [EnvWorker.remote(CloudpickleWrapper(partial(env_fn, **self.args.ENV_ARGS)), _id) for _id in
                            range(self.batch_size)]

        self.env_info, _ = ray.get(self.env_workers[0].work.remote("get_env_info", None))
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.test_return_for_model_saving = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

        self.wandb_last_log_t = 0

    def wandb_log(self, t_env):
        assert self.args.WANDB
        if (t_env - self.wandb_last_log_t) >= self.args.WANDB_LOG_INTERVAL:
            self.logger.wandb_log_stats()
            self.wandb_last_log_t = t_env

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.DEVICE)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

        # Setup the noise distribution sampler
        if self.args.NOISE_BANDIT:
            if self.args.BANDIT_POLICY:
                self.noise_distrib = enza(self.args, logger=self.logger)
            else:
                self.noise_distrib = RBandit(self.args, logger=self.logger)
        else:
           self.noise_distrib = Uniform(self.args)

        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for batch_idx in range(self.batch_size):
            self.env_workers[batch_idx].work.remote("close", None)

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Reset the envs
        obj_refs = []
        for batch_idx in range(self.batch_size):
            obj_refs.append(self.env_workers[batch_idx].work.remote("reset", None))

        data_list = []
        while obj_refs:
            ready_ids, obj_refs = ray.wait(obj_refs)
            for ready_id in ready_ids:
                data = ray.get(ready_id)
                if data is not None:
                    data_list.append(data)

        data_list.sort(key=lambda x: x[1])
        for data in data_list:
            data = data[0]
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        pre_transition_data = convert_format_to_numpy_array(pre_transition_data)
        self.batch.update(pre_transition_data, ts=0)

        # Sample the noise at the beginning of the episode
        self.noise = self.noise_distrib.sample(self.batch['state'][:,0], test_mode)

        self.batch.update({"noise": np.array(self.noise)}, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, test_uniform=False):
        try:
            self.reset(test_uniform)

            all_terminated = False
            episode_returns = [0 for _ in range(self.batch_size)]
            episode_lengths = [0 for _ in range(self.batch_size)]
            self.mac.init_hidden(batch_size=self.batch_size)
            terminated = [False for _ in range(self.batch_size)]
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            final_env_infos = []

            while True:

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch for each un-terminated env
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                cpu_actions = actions.to("cpu").detach().numpy()

                # Update the actions taken
                actions_chosen = {
                    "actions": actions.unsqueeze(1)
                }
                actions_chosen = convert_format_to_numpy_array(actions_chosen)
                self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

                obj_refs = []
                action_idx = 0
                # Send actions to each env
                for batch_idx in range(self.batch_size):
                    if batch_idx in envs_not_terminated:  # We produced actions for this env
                        if not terminated[batch_idx]:  # Only send the actions to the env if it hasn't terminated
                            obj_refs.append(self.env_workers[batch_idx].work.remote("step", cpu_actions[action_idx]))
                        action_idx += 1  # actions is not a list over every env

                # Update terminated envs after adding post_transition_data
                envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
                all_terminated = all(terminated)
                if all_terminated:
                    break

                # Post step data we will insert for the current timestep
                post_transition_data = {
                    "reward": [],
                    "terminated": []
                }
                # Data for the next step we will insert in order to select an action
                pre_transition_data = {
                    "state": [],
                    "avail_actions": [],
                    "obs": []
                }

                _terminated = terminated.copy()

                data_list = []
                while obj_refs:
                    ready_ids, obj_refs = ray.wait(obj_refs)
                    for ready_id in ready_ids:
                        data = ray.get(ready_id)
                        if data is not None:
                            data_list.append(data)

                # Receive data back for each unterminated env
                data_list.sort(key=lambda x: x[1])
                for data, idx in data_list:
                    if not terminated[idx]:
                        # Remaining data for this current timestep
                        post_transition_data["reward"].append((data["reward"],))

                        episode_returns[idx] += data["reward"]
                        episode_lengths[idx] += 1
                        if not test_mode:
                            self.env_steps_this_run += 1

                        env_terminated = False
                        if data["terminated"]:
                            final_env_infos.append(data["info"])
                        if data["terminated"] and not data["info"].get("episode_limit", False):
                            env_terminated = True
                        terminated[idx] = data["terminated"]
                        post_transition_data["terminated"].append((env_terminated,))

                        # Data for the next timestep needed to select an action
                        pre_transition_data["state"].append(data["state"])
                        pre_transition_data["avail_actions"].append(data["avail_actions"])
                        pre_transition_data["obs"].append(data["obs"])

                # Add post_transiton data into the batch
                post_transition_data = convert_format_to_numpy_array(post_transition_data)
                self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

                # Move onto the next timestep
                self.t += 1

                # Add the pre-transition data
                pre_transition_data = convert_format_to_numpy_array(pre_transition_data)
                self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

                # wandb logging
                if self.args.WANDB:
                    self.wandb_log(self.t_env)

            if not test_mode:
                self.t_env += self.env_steps_this_run

            # Get stats back for each env
            obj_refs = []
            # Send actions to each env
            for batch_idx in range(self.batch_size):
                obj_refs.append(self.env_workers[batch_idx].work.remote("get_stats", None))

            data_list = []
            while obj_refs:
                ready_ids, obj_refs = ray.wait(obj_refs)
                for ready_id in ready_ids:
                    data = ray.get(ready_id)
                    if data is not None:
                        data_list.append(data)

            cur_stats = self.test_stats if test_mode else self.train_stats
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = "test_" if test_mode else ""
            if test_uniform:
                log_prefix += "uni_"
            infos = [cur_stats] + final_env_infos
            cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
            cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
            cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

            cur_returns.extend(episode_returns)

            self._update_noise_returns(episode_returns, self.noise, final_env_infos, test_mode)
            self.noise_distrib.update_returns(self.batch['state'][:,0], self.noise, episode_returns, test_mode, self.t_env)

            n_test_runs = max(1, self.args.TEST_NEPISODE // self.batch_size) * self.batch_size
            if test_mode and (len(self.test_returns) == n_test_runs):
                self.test_return_for_model_saving = np.mean(cur_returns)
                self._log_noise_returns(test_mode, test_uniform)
                self._log(cur_returns, cur_stats, log_prefix)
            elif self.t_env - self.log_train_stats_t >= self.args.RUNNER_LOG_INTERVAL:
                self._log_noise_returns(test_mode, test_uniform)
                self._log(cur_returns, cur_stats, log_prefix)
                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

            return self.batch
        except KeyboardInterrupt:
            for batch_idx in range(self.batch_size):
                self.env_workers[batch_idx].work.remote("close", None)
            del self.env_workers

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def _update_noise_returns(self, returns, noise, stats, test_mode):
        for n, r in zip(noise, returns):
            n = int(np.argmax(n))
            if n in self.noise_returns:
                self.noise_returns[n].append(r)
            else:
                self.noise_returns[n] = [r]
        if test_mode:
            noise_won = self.noise_test_won
        else:
            noise_won = self.noise_train_won

        if stats != [] and "battle_won" in stats[0]:
            for n, info in zip(noise, stats):
                if "battle_won" not in info:
                    continue
                bw = info["battle_won"]
                n = int(np.argmax(n))
                if n in noise_won:
                    noise_won[n].append(bw)
                else:
                    noise_won[n] = [bw]

    def _log_noise_returns(self, test_mode, test_uniform):
        if test_mode:
            max_noise_return = -100000
            for n,rs in self.noise_returns.items():
                n_item = n
                r_mean = float(np.mean(rs))
                max_noise_return = max(r_mean, max_noise_return)
                self.logger.log_stat("{}_noise_test_ret_u_{:1}".format(n_item, test_uniform), r_mean, self.t_env)
            self.logger.log_stat("max_noise_test_ret_u_{:1}".format(test_uniform), max_noise_return, self.t_env)

        noise_won = self.noise_test_won
        prefix = "test"
        if test_uniform:
            prefix += "_uni"
        if not test_mode:
            noise_won = self.noise_train_won
            prefix = "train"
        if len(noise_won.keys()) > 0:
            max_test_won = 0
            for n, rs in noise_won.items():
                n_item = n #int(np.argmax(n))
                r_mean = float(np.mean(rs))
                max_test_won = max(r_mean, max_test_won)
                self.logger.log_stat("{}_noise_{}_won".format(n_item, prefix), r_mean, self.t_env)
            self.logger.log_stat("max_noise_{}_won".format(prefix), max_test_won, self.t_env)
        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

    def save_models(self, path):
        if self.args.NOISE_BANDIT:
            self.noise_distrib.save_model(path)


@ray.remote
class EnvWorker:
    def __init__(self, env_fn, worker_id):
        self.env_fn = env_fn
        self.env = env_fn.x()
        self.worker_id = worker_id

    def work(self, cmd, data):
        try:
            if cmd == "step":
                actions = data
                # Take a step in the environment
                reward, terminated, env_info = self.env.step(actions)
                # Return the observations, avail_actions and state to make the next action
                state = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                obs = self.env.get_obs()
                return {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info
                }, self.worker_id
            elif cmd == "reset":
                self.env.reset()
                return {
                    "state": self.env.get_state(),
                    "avail_actions": self.env.get_avail_actions(),
                    "obs": self.env.get_obs()
                }, self.worker_id
            elif cmd == "close":
                self.env.close()
            elif cmd == "get_env_info":
                return self.env.get_env_info(), self.worker_id
            elif cmd == "get_stats":
                return self.env.get_stats(), self.worker_id
            else:
                raise NotImplementedError
        except Exception as e:
            print("ENV Exception: ", e)
            del self.env
            self.env = self.env_fn.x()


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

