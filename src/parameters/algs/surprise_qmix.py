class SURPRISEQMIX:
    ALGORITHM = 'surprise_qmix'

    # use util epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"

    RUNNER = "surprise_parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    OBS_LAST_ACTION = True
    OBS_AGENT_ID = True
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "surprise_q_learner"
    DOUBLE_Q = True
    RNN_HIDDEN_DIM = 64
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64
    REWARD_SCALE = 1.0

    # exploration parameters
    AGENT = "intrinsic_rnn"
    AE_EXPLORATION = True
    EXP_MAC = "surprise_mac"
    EXP_AGENT = "surprise_rnn"
    BETA = 0.25
    EXP_LR = 0.0005
    MAX_INTRINSIC_REWARD = 1.0
    USE_INTRINSIC_MAC = True
    INDIVIDUAL_UPDATE = True

    # vae
    VAE_HIDDEN_DIM = 64
    LATENT_DIM = 32
    HISTORY_ENCODING_DIM = 36
    OBS_ENCODING_DIM = 32
    ACTION_ENCODING_DIM = 4

    # self-attention
    N_HEADS = 8
    MODEL_DIM = 512
