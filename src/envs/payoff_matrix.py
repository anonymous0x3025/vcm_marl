import numpy as np
from envs.multiagentenv import MultiAgentEnv


# K-step payoff matrix game
# K represents a max length of an episode
class PayOffMatrix(MultiAgentEnv):
    def __init__(self, map_name='64_step', k=64, sparse_reward=False):
        self.map_name = map_name
        self.k = k
        self.sparse_reward = sparse_reward
        self.state_num = 0
        self.info = {'n_episodes': 0, 'ep_length': self.state_num}

        value_list = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        self.n_agents = 2
        self.episode_limit = self.k
        self.path_len = 4
        self.N = 2
        self.n_actions = 2
        self.matrix = np.array(value_list).reshape(4, self.N, self.N)
        self.states = np.eye(self.k+1)
        self.matrix_index = 0
        self.done = False

    def reset(self):
        self.state_num = 0
        self.matrix_index = 0
        self.done = False
        return [self.states[self.state_num], self.states[self.state_num]], self.states[self.state_num]

    def step(self, actions):
        joint_value = self.matrix[self.matrix_index][actions[0]][actions[1]]
        reward = 0

        if self.state_num >= self.episode_limit - 1:
            self.done = True
            if self.sparse_reward:
                reward = 10
        else:
            if joint_value == 0:
                self.done = True
        if joint_value > 0:
            self.state_num += 1
            self.matrix_index = self.state_num % self.path_len
            if not self.sparse_reward:
                reward = joint_value
        self.info['ep_length'] = 0
        return [reward, self.done, self.info]

    def get_obs(self):
        return [self.states[self.state_num], self.states[self.state_num]]

    def get_state(self):
        return self.states[self.state_num]

    def get_avail_actions(self):
        return np.array([np.ones(self.get_total_actions()) for _ in range(self.n_agents)])

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.get_total_actions())

    def get_state_size(self):
        return len(self.states)

    def get_obs_size(self):
        return len(self.states)

    def get_total_actions(self):
        return self.n_actions

    def get_stats(self):
        stats = {}
        return stats

    def close(self):
        pass