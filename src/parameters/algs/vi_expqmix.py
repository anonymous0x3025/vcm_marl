class VIEXPQMIX:
    ALGORITHM = 'vi_exp_qmix'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"

    RUNNER = "parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "vi_exp_q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # vae
    KL_TO_GAUSS_PRIOR = False
    VAE_HIDDEN_DIM = 64
    LATENT_DIM = 64
    CLIP_MIN_INT_REWARD = 0
    CLIP_MAX_INT_REWARD = 5

    # exploration parameters
    BETA = 1.0
    N_STEP = 1
