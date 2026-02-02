import os
src_path = os.getcwd()
replay_path = os.path.join(src_path, "results", "replays")


class SC2:
    ENV = 'sc2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
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

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000
    # T_MAX = 1050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000

###################################################################################
###################################################################################
# sparse maps
class SC2_sparse_3m(SC2):
    MAP = '3m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': True,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000

    # BETA = 0.1
    #
    # T_MAX = 1050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000

###################################################################################
###################################################################################

###################################################################################
###################################################################################
# tset maps
class SC2_3m(SC2):
    MAP = '3m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000

    # BETA = 0.1
    #
    # T_MAX = 205000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 20000

###################################################################################
###################################################################################

###################################################################################
###################################################################################
# easy maps
class SC2_2s_vs_1sc(SC2):
    MAP = '2s_vs_1sc'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2s_vs_1sc",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000

    # BETA = 0.1
    #
    # T_MAX = 2050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000


class SC2_2s3z(SC2):
    MAP = '2s3z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "25m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000

    # BETA = 0.1
    #
    # T_MAX = 2050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000


class SC2_3s5z(SC2):
    MAP = '3s5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s5z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000

    # BETA = 0.1
    #
    # T_MAX = 2050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000


class SC2_1c3s5z(SC2):
    MAP = '1c3s5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "1c3s5z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000

    # BETA = 0.1
    #
    # T_MAX = 2050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000


class SC2_10m_vs_11m(SC2):
    MAP = '10m_vs_11m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10m_vs_11m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 3000


###################################################################################
###################################################################################

###################################################################################
###################################################################################
# hard maps
class SC2_2c_vs_64zg(SC2):
    MAP = '2c_vs_64zg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2c_vs_64zg",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000
    # BETA = 0.1
    #
    # T_MAX = 5050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 250000


class SC2_bane_vs_bane(SC2):
    MAP = 'bane_vs_bane'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "bane_vs_bane",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 5050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 250000


class SC2_5m_vs_6m(SC2):
    MAP = '5m_vs_6m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "5m_vs_6m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5#0.25

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 5050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 250000


class SC2_3s_vs_5z(SC2):
    MAP = '3s_vs_5z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s_vs_5z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.25

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 5050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 250000

###################################################################################
###################################################################################

###################################################################################
###################################################################################
# super hard maps
class SC2_3s5z_vs_3s6z(SC2):
    MAP = '3s5z_vs_3s6z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s5z_vs_3s6z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 12050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 600000


class SC2_6h_vs_8z(SC2):
    MAP = '6h_vs_8z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "6h_vs_8z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 3.0

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 12050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 600000


class SC2_27m_vs_30m(SC2):
    MAP = '27m_vs_30m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "27m_vs_30m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 12050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 600000


class SC2_MMM2(SC2):
    MAP = 'MMM2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "MMM2",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.1

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 12050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 600000


class SC2_Corridor(SC2):
    MAP = 'corridor'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "corridor",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000

    # BETA = 0.1
    #
    # T_MAX = 12050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 600000
###################################################################################
###################################################################################

###################################################################################
###################################################################################
# additional maps
class SC2_8m(SC2):
    MAP = '8m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "8m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_25m(SC2):
    MAP = '25m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "25m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_MMM(SC2):
    MAP = 'MMM'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "MMM",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_8m_vs_9m(SC2):
    MAP = '8m_vs_9m'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "8m_vs_9m",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_2m_vs_1z(SC2):
    MAP = '2m_vs_1z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "2m_vs_1z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_3s_vs_3z(SC2):
    MAP = '3s_vs_3z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s_vs_3z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_3s_vs_4z(SC2):
    MAP = '3s_vs_4z'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "3s_vs_4z",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2_so_many_baneling(SC2):
    MAP = 'so_many_baneling'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "so_many_baneling",
        'move_amount': 2,
        'obs_all_health': True,
        'obs_instead_of_state': False,
        'obs_last_action': False,
        'obs_own_health': True,
        'obs_pathing_grid': False,
        'obs_terrain_height': False,
        'obs_timestep_number': False,
        'reward_death_value': 10,
        'reward_defeat': 0,
        'reward_negative_scale': 0.5,
        'reward_only_positive': True,
        'reward_scale': True,
        'reward_scale_rate': 20,
        'reward_sparse': False,
        'reward_win': 200,
        'replay_dir': replay_path,
        'replay_prefix': "",
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'seed': None,
        'heuristic_ai': False,
        'heuristic_rest': False,
        'debug': False
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 80050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000
###################################################################################
###################################################################################