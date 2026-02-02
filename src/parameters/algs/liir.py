class LIIR:
    ALGORITHM = 'liir'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "multinomial"

    RUNNER = "parallel"

    BUFFER_SIZE = 32
    START_TRAINING_EPISODE = 1

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "pi_logits"
    LEARNER = "q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    OBS_LAST_ACTION = False

    CRITIC_LR = 0.0005
    TD_LAMBDA = 0.8
