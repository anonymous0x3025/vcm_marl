class PressurePlate:
    ENV = 'pressureplate'
    MAP = 'linear-4p' # 'linear-5p', 'linear-6p'
    EPISODE_LIMIT = 500

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    TEST_INTERVAL = 1000
    LOG_INTERVAL = 1000
    RUNNER_LOG_INTERVAL = 1000
    LEARNER_LOG_INTERVAL = 1000

    TRAIN_STEP_MAX = 30050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 0.5
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 1500

    # T_MAX = 3050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 150000

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 1.0
    INPUT_NOISE_CLIP = 2.0
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 30000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class PressurePlateLinear4P(PressurePlate):
    ENV = 'pressureplate'
    MAP = 'linear-4p'
    EPISODE_LIMIT = 400

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 30050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 1500

    # T_MAX = 12050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 600000


class PressurePlateLinear5P(PressurePlate):
    ENV = 'pressureplate'
    MAP = 'linear-5p'
    EPISODE_LIMIT = 500

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 30050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 2000

    # BETA = 0.1
    #
    # T_MAX = 15050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 750000


class PressurePlateLinear6P(PressurePlate):
    ENV = 'pressureplate'
    MAP = 'linear-6p'
    EPISODE_LIMIT = 600

    ENV_ARGS = {
        'map_name': MAP,
        'episode_limit': EPISODE_LIMIT
    }

    BETA = 0.5

    TRAIN_STEP_MAX = 30050
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 2000
    # BETA = 0.1
    #
    # T_MAX = 20050000
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 1000000
