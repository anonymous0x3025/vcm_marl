import ray
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
from utils.rl_utils import convert_format_to_numpy_array


class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.BATCH_SIZE_RUN

        env_fn = env_REGISTRY[self.args.ENV]

        self.env_workers = [EnvWorker.remote(CloudpickleWrapper(partial(env_fn, **self.args.ENV_ARGS)), _id) for _id in range(self.batch_size)]

        self.env_info, _ = ray.get(self.env_workers[0].work.remote("get_env_info", None))
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.test_return_for_model_saving = 0

        self.log_train_stats_t = -100000

        self.wandb_last_log_t = 0

    def wandb_log(self, t_env):
        assert self.args.WANDB
        if (t_env - self.wandb_last_log_t) >= self.args.WANDB_LOG_INTERVAL:
            self.logger.wandb_log_stats()
            self.wandb_last_log_t = t_env

    def setup(self, scheme, groups, preprocess, mac, hidden_mac=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.DEVICE)
        self.mac = mac
        if self.args.HIDDEN_POLICY:
            self.hidden_mac = hidden_mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        for batch_idx in range(self.batch_size):
            self.env_workers[batch_idx].work.remote("save_replay", None)

    def close_env(self):
        for batch_idx in range(self.batch_size):
            self.env_workers[batch_idx].work.remote("close", None)

    def reset(self):
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

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        try:
            self.reset()

            all_terminated = False
            episode_returns = [0 for _ in range(self.batch_size)]
            episode_lengths = [0 for _ in range(self.batch_size)]
            # if test_mode and self.args.HIDDEN_POLICY:
            #     self.hidden_mac.init_hidden(batch_size=self.batch_size)
            # else:
            self.mac.init_hidden(batch_size=self.batch_size)
            terminated = [False for _ in range(self.batch_size)]
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

            while True:
                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch for each un-terminated env
                # if test_mode and self.args.HIDDEN_POLICY:
                #     actions = self.hidden_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                # else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)

                if type(actions) == tuple:
                    actions = actions[0]

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

                # Update envs_not_terminated
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
                    # Remaining data for this current timestep
                    adjusted_reward = self.args.EXTRINSIC_REWARD_WEIGHT * data["reward"]
                    post_transition_data["reward"].append((adjusted_reward,))

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

            # Receive data back for each unterminated env
            data_list.sort(key=lambda x: x[1])
            env_stats = []
            for data in data_list:
                env_stats.append(data[0])

            cur_stats = self.test_stats if test_mode else self.train_stats
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = "test_" if test_mode else ""
            infos = [cur_stats] + final_env_infos
            cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
            cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
            cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
            if self.test_stats is not {}:
                cur_stats["ep_length"] = cur_stats.get("ep_length", 0)
            else:
                cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

            cur_returns.extend(episode_returns)

            n_test_runs = max(1, self.args.TEST_NEPISODE // self.batch_size) * self.batch_size
            if test_mode and (len(self.test_returns) == n_test_runs):
                self.test_return_for_model_saving = np.mean(cur_returns)
                self._log(cur_returns, cur_stats, log_prefix)
            elif self.t_env - self.log_train_stats_t >= self.args.RUNNER_LOG_INTERVAL:
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
            elif cmd == "save_replay":
                self.env.save_replay()
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
