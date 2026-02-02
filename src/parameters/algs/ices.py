class ICESQMIX:
    ALGORITHM = 'ices'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_expl"

    RUNNER = "parallel"

    BUFFER_SIZE = 5000

    EPSILON_UPDATE_STANDARD = 'steps'

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT = "n_rnn"
    MAC = "ices_n_mac"
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "ices_nq_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    optimizer = 'adam'

    # rnn layer normalization
    use_layer_norm = False

    # orthogonal init for DNN
    use_orthogonal = False
    gain = 0.01
    t_max = 8050000

    # Priority experience replay
    use_per = False
    per_alpha = 0.6
    per_beta = 0.4
    return_priority = False

    lr = 0.001
    embedding_dim = 4
    hidden_dim = 64
    z_dim = 16
    pred_s_len = 1
    world_bl_lr = 0.0001
    world_lr = 0.0001
    world_clip_param = 0.1
    world_gamma = 0.01
    weight_decay = 0
    int_lr = 0.01
    int_c_lr = 0.01
    int_ratio = 0.1
    int_finish = 0.1
    int_ent_coef = 0.1
    step_penalty = -0.02
    td_lambda = 0.6
    norm_s = True
    optim_alpha = 0.99
    optim_eps = 0.00001
    learner_log_interval = 2000
