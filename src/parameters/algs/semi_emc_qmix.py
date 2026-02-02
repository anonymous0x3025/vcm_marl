class SEMIEMCQMIX:
    ALGORITHM = 'semi_emc_qmix'

    # use epsilon greedy action selector
    ACTION_SELECTOR = "epsilon_greedy"

    RUNNER = "parallel"

    BUFFER_SIZE = 5000

    # update the target network every {} episodes
    TARGET_UPDATE_INTERVAL = 200

    # use the Q_Learner to train
    AGENT_OUTPUT_TYPE = "q"
    LEARNER = "semi_emc_q_learner"
    DOUBLE_Q = True
    MIXER = "qmix"
    MIXING_EMBED_DIM = 32
    HYPERNET_LAYERS = 2
    HYPERNET_EMBED = 64

    # intrinsic reward weighting
    RHO = 0.5
    BETA = 1.0
