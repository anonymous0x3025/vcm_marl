import os
src_path = os.getcwd()
replay_path = os.path.join(src_path, "results", "replays")

class SC2V2:
    ENV = 'sc2_v2'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_zerg",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions':12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config':{
            'n_units': 5,
            'n_enemies': 5,
            'team_gen':{
                'dist_type': "weighted_teams",
                'unit_types':[
                    "zergling",
                    "baneling",
                    "hydralisk"
                ],
                'weights': [0.45, 0.1, 0.45],
                'exception_unit_types': ["baneling"],
                'observe': True
            },
            'start_positions':{
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
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

    BETA = 1.0

    TRAIN_STEP_MAX = 30050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000
    # T_MAX = 1050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 100000

###################################################################################
###################################################################################
# Zerg
class SC2V2_10gen_zerg(SC2V2):
    MAP = '10gen_zerg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_zerg",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 5,
            'n_enemies': 5,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "zergling",
                    "baneling",
                    "hydralisk"
                ],
                'weights': [0.45, 0.1, 0.45],
                'exception_unit_types': ["baneling"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_40gen_zerg(SC2V2):
    MAP = '40gen_zerg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_zerg",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 20,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "zergling",
                    "baneling",
                    "hydralisk"
                ],
                'weights': [0.45, 0.1, 0.45],
                'exception_unit_types': ["baneling"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_43gen_zerg(SC2V2):
    MAP = '43gen_zerg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_zerg",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 23,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "zergling",
                    "baneling",
                    "hydralisk"
                ],
                'weights': [0.45, 0.1, 0.45],
                'exception_unit_types': ["baneling"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_46gen_zerg(SC2V2):
    MAP = '46gen_zerg'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_zerg",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 26,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "zergling",
                    "baneling",
                    "hydralisk"
                ],
                'weights': [0.45, 0.1, 0.45],
                'exception_unit_types': ["baneling"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000
###################################################################################
###################################################################################

###################################################################################
###################################################################################
# Protoss
class SC2V2_10gen_protoss(SC2V2):
    MAP = '10gen_protoss'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_protoss",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 5,
            'n_enemies': 5,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "stalker",
                    "zealot",
                    "colossus"
                ],
                'weights': [0.45, 0.45, 0.1],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_40gen_protoss(SC2V2):
    MAP = '40gen_protoss'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_protoss",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 20,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "stalker",
                    "zealot",
                    "colossus"
                ],
                'weights': [0.45, 0.45, 0.1],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_43gen_protoss(SC2V2):
    MAP = '43gen_protoss'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_protoss",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 23,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "stalker",
                    "zealot",
                    "colossus"
                ],
                'weights': [0.45, 0.45, 0.1],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_46gen_protoss(SC2V2):
    MAP = '46gen_protoss'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_protoss",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 26,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "stalker",
                    "zealot",
                    "colossus"
                ],
                'weights': [0.45, 0.45, 0.1],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000
###################################################################################
###################################################################################

###################################################################################
###################################################################################
# Terran
class SC2V2_10gen_terran(SC2V2):
    MAP = '10gen_terran'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_terran",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 5,
            'n_enemies': 5,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "marine",
                    "marauder",
                    "medivac"
                ],
                'weights': [0.45, 0.45, 0.1],
                'exception_unit_types': ["medivac"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_40gen_terran(SC2V2):
    MAP = '40gen_terran'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_terran",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 20,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "marine",
                    "marauder",
                    "medivac"
                ],
                'weights': [0.45, 0.45, 0.1],
                'exception_unit_types': ["medivac"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_43gen_terran(SC2V2):
    MAP = '43gen_terran'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_terran",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 23,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "marine",
                    "marauder",
                    "medivac"
                ],
                'weights': [0.45, 0.45, 0.1],
                'exception_unit_types': ["medivac"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000


class SC2V2_46gen_terran(SC2V2):
    MAP = '46gen_terran'
    ENV_ARGS = {
        'continuing_episode': False,
        'difficulty': "7",
        'game_version': "latest",
        'map_name': "10gen_terran",
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
        'conic_fov': False,
        'use_unit_ranges': True,
        'min_attack_range': 2,
        'num_fov_actions': 12,
        'obs_own_pos': True,
        'fully_observable': False,
        'capability_config': {
            'n_units': 20,
            'n_enemies': 26,
            'team_gen': {
                'dist_type': "weighted_teams",
                'unit_types': [
                    "marine",
                    "marauder",
                    "medivac"
                ],
                'weights': [0.45, 0.45, 0.1],
                'exception_unit_types': ["medivac"],
                'observe': True
            },
            'start_positions': {
                'dist_type': "surrounded_and_reflect",
                'p': 0.5,
                'map_x': 32,
                'map_y': 32
            }
        },
        'state_last_action': True,
        'state_timestep_number': False,
        'step_mul': 8,
        'heuristic_ai': False,
        'debug': False
    }

    BETA = 1.0

    TRAIN_STEP_MAX = 100050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 5000
###################################################################################
###################################################################################