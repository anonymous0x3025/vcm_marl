from .parameters_preamble import *


class MultiParameters:
    param1 = VIIntrinsicQmixPom16stepSparse()
    param1.DEVICE = 'cpu' # 'mps', 'cuda:0', 'cpu'
    param1.BUFFER_CPU_ONLY = True
    param1.WANDB = True
    param1.BATCH_SIZE_RUN = 2
    param1.RUNS = 2

    param2 = VIIntrinsicQmixPom32stepSparse()
    param2.DEVICE = 'cpu' # 'mps', 'cuda:0', 'cpu'
    param2.BUFFER_CPU_ONLY = True
    param2.WANDB = True
    param1.BATCH_SIZE_RUN = 2
    param2.RUNS = 2

    params_list = [param1, param2]


super_hard_maps = ["3s5z_vs_3s6z", "6h_vs_8z", "27m_vs_30m", "MMM2", "corridor"]
hard_maps = ["3s_vs_5z", "bane_vs_bane", "5m_vs_6m", "2c_vs_64zg"]
