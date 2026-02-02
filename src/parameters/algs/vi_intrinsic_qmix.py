class VIINTRINSICQMIX:
    ALGORITHM = 'vi_intrinsic_qmix'

    # use util epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"

    RUNNER = "intrinsic_parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    INTRINSIC_TARGET_UPDATE_INTERVAL = 50
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    OBS_LAST_ACTION = True
    OBS_AGENT_ID = True
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "vi_intrinsic_q_learner"
    DOUBLE_Q = True
    RNN_HIDDEN_DIM = 64
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64
    REWARD_SCALE = 1.0

    # exploration parameters
    AGENT = "intrinsic_rnn_belief"
    AE_EXPLORATION = True
    EXP_MAC = "vi_intrinsic_mac"
    EXP_AGENT = "vae_intrinsic_rnn"
    INTRINSIC_LR = 0.0008
    INTRINSIC_GRAD_NORM_CLIP = 3.0
    CLIP_INTRINSIC_REWARD = 1.0
    # ablation ###########################
    USE_INT_REWARD = True
    USE_ATTENTION = True
    ######################################
    SAMPLE_EMBEDDINGS = False
    INDIVIDUAL_UPDATE = False
    USE_INTRINSIC_MAC = True
    INTRINSIC_MAC = "intrinsic_critic_mac"
    INTRINSIC_AGENT = "intrinsic_critic_rnn"
    INT_REWARD_SCALE_MAX = 1.0
    INT_REWARD_SCALE_MIN = 0.0
    # INITIATION_GIVING_INT_REWARD = 1000

    # vae
    KL_TO_GAUSS_PRIOR = False
    VAE_HIDDEN_DIM = 64
    LATENT_DIM = 32
    OBS_ENCODING_DIM = 32
    STATE_ENCODING_DIM = 32
    ACTION_ENCODING_DIM = 4

    # self-attention
    N_HEADS = 4
    MODEL_DIM = 256
