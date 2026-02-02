import os
src_path = os.getcwd()
replay_path = os.path.join(src_path, "results", "replays")


class STAGHUNT:
    ENV = 'stag_hunt'
    MAP = 'default'
    ENV_ARGS = {
        'capture_action': True,
        'n_agents': 8,
        'n_stags': 8,
        'n_hare': 0,
        'miscapture_punishment': -2.0,
        'agent_obs': [2, 2],      # (radius-1) of the agent's observation, e.g., [0, 0] observes only one pixel
        'agent_move_block': [0,1,2],   # by which entities is an agent's move blocked (0=agents, 1=stags, 2=hare)
        'capture_conditions': [0, 1],  # number of free fields available to [stag, hare] to be captured
        'capture_action_conditions': [2, 1],  # number of agents that have to simultaneously execute "catch" action
        'capture_freezes': True,   # whether capturing any prey freezes the participating agents (True) or not (False)
        'capture_terminal': False, # whether capturing any prey ends the episode
        'directed_observations': False,   # Agents observe square around them (False) or a cone in the direction of the last action (True).
        'directed_cone_narrow': True,     # Whether the diagonal is excluded from the directed_observation cone (True)
        'directed_exta_actions': True,    # Whether the observation cone is controlled by movement (False) or actions (True)
        'episode_limit': 200,      # maximum number of time steps per episode
        'intersection_global_view': False, # intersection specific (MACKRL)
        'intersection_unknown': False,     # intersection specific (MACKRL)
        'mountain_slope': 0.3,     # probability that an "up" action will not be executed (stag_hunt = 0.0)
        'mountain_spawn': False,   # whether prey spawns in their preferred habitat (True) or randomly (False)
        'mountain_agent_row': -1,  # the row in which the agents are spawned (0 is top). Negative values spawn agents randomly.
        'observe_state': False,    # whether an observation is only partial (False) or central including agent position (True)
        'observe_walls': False,    # observe walls as an extra feature (only for state_as_list=False and toroidal=False)
        'observe_ids': False,      # observe agent ID, instead of agent presence (only for state_as_list=False)
        'observe_one_hot': False,  # observe agent ID as one-hot vector (only for observer_ids=True)
        'p_stags_rest': 0.0,       # probability that a stag will not move (at each time step)
        'p_hare_rest': 0.0,        # probability that a hare will not move (at each time step)
        'prevent_cannibalism': True,   # If set to False, prey can be captured by other prey (witch is rewarding)
        'print_caught_prey': False,    # debug messages about caught prey and finished episodes
        'print_frozen_agents': False,  # debug messages about frozen agents after some prey has been caught
        'random_ghosts': False,    # If True, prey turns into ghosts randomly (neg. reward), indicated by a corner-feature
        'random_ghosts_prob': 0.5, # Probability that prey turns into ghost
        'random_ghosts_mul': -1,   # Catching ghosts incurs a reward/punishment of random_ghost_mul*reward
        'random_ghosts_indicator': False,  # If True, the indicator for ghosts is in a different corner every episode
        'remove_frozen': True,     # whether frozen agents are removed (True) or still present in the world (False)
        'reward_hare': 1,          # reward for capturing a hare
        'reward_stag': 10,         # reward for capturing a stag
        'reward_collision': 0,     # reward (or punishment) for colliding with other agents
        'reward_time': 0,          # reward (or punishment) given at each time step
        'state_as_graph': False,   # whether the state is a list of entities (True) or the entire grid (False
        'toroidal': False,         # whether the world is bounded (False) or toroidal (True)
        'world_shape': [10, 10],   # the shape of the grid-world [height, width]
    }

    TEST_GREEDY = True
    TEST_NEPISODE = 32
    TEST_INTERVAL = 2500
    LOG_INTERVAL = 2500
    RUNNER_LOG_INTERVAL = 2500
    LEARNER_LOG_INTERVAL = 2500

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 0.2
    INPUT_NOISE_CLIP = 0.5
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 100000

    EXTRINSIC_REWARD_WEIGHT = 1.0

    BETA = 1.0
    RHO = 0.5

    TRAIN_STEP_MAX = 10050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 2000
