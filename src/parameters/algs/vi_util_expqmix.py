class VIUTILEXPQMIX:
    ALGORITHM = 'vi_util_exp_qmix'

    # use util epsilon greedy action selector
    ACTION_SELECTOR = "util_epsilon_greedy"

    RUNNER = "util_parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200
    EXP_TARGET_UPDATE_INTERVAL = 5

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "vi_util_exp_q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # exploration parameters
    AGENT = "util_rnn"
    AE_EXPLORATION = True
    EXP_MAC = "vi_util_exp_mac"
    EXP_AGENT = "vae_rnn"
    BETA = 1.0
    UTIL_LR = 0.0005  # Learning rate for exp. util. agents

    # vae
    KL_TO_GAUSS_PRIOR = False
    VAE_HIDDEN_DIM = 64
    LATENT_DIM = 64
    OBS_ENCODING_DIM = 64
    ACTION_ENCODING_DIM = 16
    VAE_LR = 0.001  # Learning rate for vae
