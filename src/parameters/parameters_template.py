from .parameters_preamble import *


class Parameters(VIIntrinsicQmixSc3M):
    DEVICE = 'cpu' # 'mps', 'cuda:0', 'cpu'
    BUFFER_CPU_ONLY = True
    WANDB = False
    WANDB_ENTITY = 'user_entity'
    WANDB_PROJECT_NAME = 'marl'
