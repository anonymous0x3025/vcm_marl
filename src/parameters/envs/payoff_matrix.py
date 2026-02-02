class PayoffMatrix:
    ENV = 'payoff_matrix'
    MAP = '64_step'
    K = 64
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    T_MAX = 20000

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 2000

    TEST_INTERVAL = 1000
    LOG_INTERVAL = 1000
    RUNNER_LOG_INTERVAL = 1000
    LEARNER_LOG_INTERVAL = 1000

    TRAIN_STEP_MAX = 100
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    ENCODER_RNN = True
    USE_INPUT_NOISE = False
    INPUT_NOISE = 1.0
    INPUT_NOISE_CLIP = 2.0
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.1
    INPUT_NOISE_DECAY_ANNEAL_TIME = 30000

    EXTRINSIC_REWARD_WEIGHT = 1.0


class PayoffMatrix64Step(PayoffMatrix):
    MAP = '64_step'
    K = 64
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    TRAIN_STEP_MAX = 510
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 100
    BETA = 1.0

    # T_MAX = 20500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 1000


class PayoffMatrix128Step(PayoffMatrix):
    MAP = '128_step'
    K = 128
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    TRAIN_STEP_MAX = 1010
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 200
    BETA = 1.0

    # T_MAX = 30500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 1500


class PayoffMatrix256Step(PayoffMatrix):
    MAP = '256_step'
    K = 256
    ENV_ARGS = {
        'map_name': MAP,
        'k': K
    }

    TRAIN_STEP_MAX = 2010
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 400
    BETA = 1.0

    # T_MAX = 50500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 2500

class PayoffMatrix4StepSparse(PayoffMatrix):
    MAP = '4_step_sparse'
    K = 4
    ENV_ARGS = {
        'map_name': MAP,
        'k': K,
        'sparse_reward': True
    }

    TRAIN_STEP_MAX = 510
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 100
    BETA = 1.0

    # T_MAX = 10500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 500

class PayoffMatrix8StepSparse(PayoffMatrix):
    MAP = '8_step_sparse'
    K = 8
    ENV_ARGS = {
        'map_name': MAP,
        'k': K,
        'sparse_reward': True
    }

    TRAIN_STEP_MAX = 1010
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 200
    BETA = 1.0

    # T_MAX = 20500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 1000

class PayoffMatrix16StepSparse(PayoffMatrix):
    MAP = '16_step_sparse'
    K = 16
    ENV_ARGS = {
        'map_name': MAP,
        'k': K,
        'sparse_reward': True
    }

    TRAIN_STEP_MAX = 2010
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 400
    BETA = 1.0

    # T_MAX = 30500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 1500

class PayoffMatrix32StepSparse(PayoffMatrix):
    MAP = '32_step_sparse'
    K = 32
    ENV_ARGS = {
        'map_name': MAP,
        'k': K,
        'sparse_reward': True
    }

    TRAIN_STEP_MAX = 3010
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.1
    EPSILON_ANNEAL_TIME = 600
    BETA = 1.0

    # T_MAX = 50500
    #
    # EPSILON_START = 1.0
    # EPSILON_FINISH = 0.05
    # EPSILON_ANNEAL_TIME = 2500