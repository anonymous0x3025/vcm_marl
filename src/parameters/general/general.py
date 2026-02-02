class General:
    # Wandb
    WANDB = False
    WANDB_ENTITY = 'jubong'
    WANDB_PROJECT_NAME = 'EXP_MARL'
    WANDB_LOG_INTERVAL = 100

    ALGORITHM = 'qmix'

    RUNS = 1
    NUM_CPUS = 10
    NUM_GPUS = 1

    # use exploration technique
    AE_EXPLORATION = False

    # --- pymarl options ---
    RUNNER = "episode"  # Runs 1 env for an episode
    MAC = "basic_mac"  # Basic controller
    ENV = "sc2"  # Environment name
    ENV_ARGS = {}  # Arguments for the environment
    BATCH_SIZE_RUN = 1  # Number of environments to run in parallel
    TEST_NEPISODE = 20  # Number of episodes to test for
    TEST_INTERVAL = 2000  # Test after {} timesteps have passed
    TEST_GREEDY = True  # Use greedy evaluation (if False, will set epsilon floor to 0
    LOG_INTERVAL = 2000  # Log summary of stats after every {} timesteps
    RUNNER_LOG_INTERVAL = 2000  # Log runner stats (not test stats) every {} timesteps
    LEARNER_LOG_INTERVAL = 2000  # Log training stats every {} timesteps
    TRAIN_STEP_MAX = 1000  # Stop running after this many timesteps
    INT_REWARD_SCALE_ANNEAL_TIME = TRAIN_STEP_MAX
    INT_REWARD_SCALE_MAX = 1.0
    INT_REWARD_SCALE_MIN = 0.0
    DEVICE = 'cpu' # 'mps', 'cuda:0', 'cpu' // Use gpu by default unless it isn't available
    BUFFER_CPU_ONLY = True  # If true we won't keep all of the replay buffer in vram

    # --- Logging options ---
    USE_TENSORBOARD = False  # Log results to tensorboard
    SAVE_MODEL = False  # Save the models to disk
    LOAD_MODEL = False  # Load the models from disk
    EVALUATE = False  # Evaluate model for test_nepisode episodes and quit (no training)
    SAVE_REPLAY = False  # Saving the replay of the model loaded from checkpoint_path
    LOCAL_RESULTS_PATH = "results"  # Path for local results

    # --- RL hyperparameters ---
    GAMMA = 0.99
    BATCH_SIZE = 32  # Number of episodes to train on
    BATCH_SEQUENCE = False
    BUFFER_SIZE = 5000  # Size of the replay buffer
    START_TRAINING_EPISODE = 32
    REWARD_SCALE = 1.0
    LR = 0.0005  # Learning rate for agents
    CRITIC_LR = 0.0005  # Learning rate for critics
    OPTIM_ALPHA = 0.99  # RMSProp alpha
    OPTIM_EPS = 0.00001  # RMSProp epsilon
    GRAD_NORM_CLIP = 5  # Reduce magnitude of gradients above this L2 norm

    # --- Agent parameters ---
    AGENT = "rnn"  # Default rnn agent
    ACTION_SPACE = "discrete"
    EPSILON_UPDATE_STANDARD = "train_steps"     # train_steps, steps
    RNN_HIDDEN_DIM = 64  # Size of hidden state for default rnn agent
    OBS_AGENT_ID = True  # Include the agent's one_hot id in the observation
    OBS_LAST_ACTION = True  # Include the agent's last action (one_hot) in the observation

    # --- Experiment running params ---
    REPEAT_ID = 1
    LABEL = "default_label"

    # --- EXP ---
    ENCODER_RNN = False
    SOFT_UPDATE_TAU = 0.005
    HIDDEN_POLICY = False
    USE_ATTENTION = True
    USE_INT_REWARD = True
    N_STEP = False
    BETA = 1.0
