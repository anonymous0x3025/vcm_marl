class EXPQMIX:
    ALGORITHM = 'exp_qmix'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"

    RUNNER = "parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "exp_q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # exploration parameters
    AE_EXPLORATION = True
    HIDDEN_POLICY = False
    EXP_MAC = "exp_mac"
    EXP_AGENT = "auto_encoder"
    RHO = 0.5
    BETA = 1.0
    INT_REWARD_SCALE_MAX = 1.0
    INT_REWARD_SCALE_MIN = 0.0
    ENCODER_HIDDEN_DIM = 64
    FEATURE_HIDDEN_DIM = 64
    RNN_HIDDEN_DIM = 32
    ENCODER_RNN = False

    INPUT_NOISE = 0.2
    INPUT_NOISE_CLIP = 1.0
    INPUT_NOISE_DECAY_START = 1.0
    INPUT_NOISE_DECAY_FINISH = 0.01
    INPUT_NOISE_DECAY_ANNEAL_TIME = 20000

    Q_INT_REWARD = True
