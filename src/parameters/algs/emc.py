class EMC:
    ALGORITHM = 'emc'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"

    RUNNER = "episode"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "qplex_curiosity_vdn_learner"
    DOUBLE_Q = True
    MIXER = "dmaq"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # params in Q_PLEX
    ADV_HYPERNET_LAYERS = 1
    ADV_HYPERNET_EMBED = 64
    NUM_KERNEL = 4
    IS_MINUS_ONE = True
    WEIGHTED_HEAD = True
    IS_ADV_ATTENTION = True
    IS_STOP_GRADIENT = True
    NONLINEAR = False
    STATE_BIAS = True

    SAVE_BUFFER = False

    CURIOSITY_SCALE = 0.1

    # EM
    IS_PRIORITIZED_BUFFER = False
    USE_EMDQN = True
    EMDQN_LOSS_WEIGHT = 0.01

    EMDQN_BUFFER_SIZE = 1000000
    EMDQN_LATENT_DIM = 4

    SOFT_UPDATE_TAU = 0.005
    VDN_SOFT_UPDATE = True
    PREDICT_VDN_TARGET = True
    PREDICT2_VDN_TARGET = True
    USE_QTOTAL_TD = False

    MAC = "fast_mac"
    AGENT = "rnn_fast"
    USE_INDIVIDUAL_Q: False
    INDIVIDUAL_Q_LOSS_WEIGHT: 0.01