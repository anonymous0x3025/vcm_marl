class NStepMatrix:
    ENV = "n_step_matrix"

    ENV_ARGS = {
        'env_args': {
            "steps": 1,
            "good_branches": 2
        }
    }

    TEST_NEPISODE = 32
    TEST_INTERVAL = 1000
    LOG_INTERVAL = 1000
    RUNNER_LOG_INTERVAL = 1000
    LEARNER_LOG_INTERVAL = 1000

    T_MAX = 20000

    EPSILON_START = 1.0
    EPSILON_FINISH = 0.05
    EPSILON_ANNEAL_TIME = 2000
