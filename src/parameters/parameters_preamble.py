from parameters.general.general import General

from parameters.algs.qmix import QMIX
from parameters.algs.expqmix import EXPQMIX
from parameters.algs.qtran import QTRAN
from parameters.algs.expqtran import EXPQTRAN
from parameters.algs.qplex import QPLEX
from parameters.algs.expqplex import EXPQPLEX
from parameters.algs.vi_intrinsic_qplex import VIINTRINSICQPLEX
from parameters.algs.matd3 import MATD3
from parameters.algs.expmatd3 import EXPMATD3
from parameters.algs.cw_qmix import CWQMIX
from parameters.algs.expcw_qmix import EXPCWQMIX
from parameters.algs.ow_qmix import OWQMIX
from parameters.algs.expow_qmix import EXPOWQMIX
from parameters.algs.semi_emc_qmix import SEMIEMCQMIX
from parameters.algs.rnd_qmix import RNDQMIX
from parameters.algs.icm_qmix import ICMQMIX
from parameters.algs.noise_qmix import NOISEQMIX as MAVEN
from parameters.algs.vi_expqmix import VIEXPQMIX
from parameters.algs.vi_util_expqmix import VIUTILEXPQMIX
from parameters.algs.vi_intrinsic_qmix import VIINTRINSICQMIX
from parameters.algs.emc import EMC
from parameters.algs.surprise_qmix import SURPRISEQMIX
from parameters.algs.liir import LIIR
from parameters.algs.vi_intrinsic_cw_qmix import VIINTRINSICCWQMIX
from parameters.algs.vi_intrinsic_ow_qmix import VIINTRINSICOWQMIX
from parameters.algs.vi_intrinsic_qtran import VIINTRINSICQTRAN
from parameters.algs.ices import ICESQMIX
from parameters.envs.payoff_matrix import PayoffMatrix64Step, PayoffMatrix128Step, PayoffMatrix256Step, PayoffMatrix4StepSparse, PayoffMatrix8StepSparse, PayoffMatrix16StepSparse, PayoffMatrix32StepSparse
from parameters.envs.sc2 import SC2_3m, SC2_sparse_3m, SC2_27m_vs_30m, SC2_3s_vs_5z, SC2_3s5z, SC2_2c_vs_64zg, SC2_Corridor, SC2_6h_vs_8z, SC2_5m_vs_6m, SC2_MMM2, SC2_3s5z_vs_3s6z, SC2_2s_vs_1sc, SC2_2s3z, SC2_1c3s5z, SC2_10m_vs_11m, SC2_bane_vs_bane, SC2_8m, SC2_2m_vs_1z, SC2_MMM, SC2_3s_vs_3z, SC2_3s_vs_4z, SC2_so_many_baneling, SC2_25m, SC2_8m_vs_9m
from parameters.envs.sc2_v2 import SC2V2_10gen_zerg, SC2V2_40gen_zerg, SC2V2_43gen_zerg, SC2V2_46gen_zerg, SC2V2_10gen_protoss, SC2V2_40gen_protoss, SC2V2_43gen_protoss, SC2V2_46gen_protoss, SC2V2_10gen_terran, SC2V2_40gen_terran, SC2V2_43gen_terran, SC2V2_46gen_terran
from parameters.envs.ma_mujoco import MaMujoco2AAnt, MaMujoco4AAnt, MaMujoco2AHalfcheetah, MaMujoco6AHalfcheetah, MaMujoco3AHopper, MaMujoco2AHumanoid, MaMujoco2AHumanoidStandup, MaMujocoManyAgentSwimmer, MaMujocoCoupledHalfCheetah, MaMujocoManyAgentAnt
from parameters.envs.pettingzoo import PettingZooPistonBall, PettingZooCooperativePong, PettingZooSimpleSpread, PettingZooPursuit
from parameters.envs.pressureplate import PressurePlateLinear4P, PressurePlateLinear5P, PressurePlateLinear6P
from parameters.envs.rwarehouse import RWarehouseTiny2Ag, RWarehouseSmall4Ag, RWarehouseHard6Ag
from parameters.envs.stag_hunt import STAGHUNT

from utils.param_utils import check_parameters_overlapped


# Very Important !!!
# The below rule must be observed
# 1. class definition priority (env -> alg -> general)
# 2. checking parameters priority (general -> alg -> env)

######################################################################################################
# QMIX

# Payoff-matrix Game: K-step
class QmixPom64step(PayoffMatrix64Step, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix64Step)
class QmixPom128step(PayoffMatrix128Step, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix128Step)
class QmixPom256step(PayoffMatrix256Step, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix256Step)
class QmixPom4stepSparse(PayoffMatrix4StepSparse, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix4StepSparse)
class QmixPom8stepSparse(PayoffMatrix8StepSparse, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix8StepSparse)
class QmixPom16stepSparse(PayoffMatrix16StepSparse, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix16StepSparse)
class QmixPom32stepSparse(PayoffMatrix32StepSparse, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PayoffMatrix32StepSparse)

class QmixPZPistonBall(PettingZooPistonBall, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooPistonBall)
class QmixPZCooperativePong(PettingZooCooperativePong, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooCooperativePong)
class QmixPZSimpleSpread(PettingZooSimpleSpread, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooSimpleSpread)
class QmixPZPursuit(PettingZooPursuit, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PettingZooPursuit)

class QmixPP4P(PressurePlateLinear4P, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PressurePlateLinear4P)
class QmixPP5P(PressurePlateLinear5P, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PressurePlateLinear5P)
class QmixPP6P(PressurePlateLinear6P, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, PressurePlateLinear6P)

class QmixRWTiny2Ag(RWarehouseTiny2Ag, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, RWarehouseTiny2Ag)
class QmixRWSmall4Ag(RWarehouseSmall4Ag, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, RWarehouseSmall4Ag)
class QmixRWHard6Ag(RWarehouseHard6Ag, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class QmixSc3M(SC2_3m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3m)

# StarCraft2: 3m
class QmixScSparse3M(SC2_sparse_3m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class QmixSc27Mvs30M(SC2_27m_vs_30m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class QmixSc3Svs5Z(SC2_3s_vs_5z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class QmixSc3S5Z(SC2_3s5z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class QmixSc2Cvs64ZG(SC2_2c_vs_64zg, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class QmixScCorridor(SC2_Corridor, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class QmixSc5Mvs6M(SC2_5m_vs_6m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class QmixSc6Hvs8Z(SC2_6h_vs_8z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class QmixScMMM2(SC2_MMM2, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class QmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class QmixSc2Svs1SC(SC2_2s_vs_1sc, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class QmixSc2S3Z(SC2_2s3z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class QmixSc1C3S5Z(SC2_1c3s5z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class QmixSc10Mvs11M(SC2_10m_vs_11m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class QmixScBANEvsBANE(SC2_bane_vs_bane, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_bane_vs_bane)

# StarCraft2: SC2_8m
class QmixSc8M(SC2_8m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_8m)

# StarCraft2: SC2_2m_vs_1z
class QmixSc2Mvs1Z(SC2_2m_vs_1z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_2m_vs_1z)

# StarCraft2: SC2_MMM
class QmixScMMM(SC2_MMM, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_MMM)

# StarCraft2: SC2_3s_vs_3z
class QmixSc3Svs3Z(SC2_3s_vs_3z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s_vs_3z)

# StarCraft2: SC2_3s_vs_4z
class QmixSc3Svs4Z(SC2_3s_vs_4z, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_3s_vs_4z)

# StarCraft2: SC2_so_many_baneling
class QmixScSoManyBaneling(SC2_so_many_baneling, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_so_many_baneling)

# StarCraft2: SC2_25m
class QmixSc25M(SC2_25m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_25m)

# StarCraft2: SC2_8m_vs_9m
class QmixSc8Mvs9M(SC2_8m_vs_9m, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2_8m_vs_9m)

# StarCraft2-v2: SC2V2
class QmixScv2_10GenZerg(SC2V2_10gen_zerg, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_10gen_zerg)

class QmixScv2_40GenZerg(SC2V2_40gen_zerg, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_40gen_zerg)

class QmixScv2_43GenZerg(SC2V2_43gen_zerg, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_43gen_zerg)

class QmixScv2_46GenZerg(SC2V2_46gen_zerg, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_46gen_zerg)

class QmixScv2_10GenProtoss(SC2V2_10gen_protoss, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_10gen_protoss)

class QmixScv2_40GenProtoss(SC2V2_40gen_protoss, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_40gen_protoss)

class QmixScv2_43GenProtoss(SC2V2_43gen_protoss, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_43gen_protoss)

class QmixScv2_46GenProtoss(SC2V2_46gen_protoss, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_46gen_protoss)

class QmixScv2_10GenTerran(SC2V2_10gen_terran, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_10gen_terran)

class QmixScv2_40GenTerran(SC2V2_40gen_terran, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_40gen_terran)

class QmixScv2_43GenTerran(SC2V2_43gen_terran, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_43gen_terran)

class QmixScv2_46GenTerran(SC2V2_46gen_terran, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, SC2V2_46gen_terran)

# STAG HUNT
class QmixStagHunt(STAGHUNT, QMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QMIX, STAGHUNT)
######################################################################################################


######################################################################################################
# SURPRISEQMIX

# Payoff-matrix Game: K-step
class SupriseQmixPom64step(PayoffMatrix64Step, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PayoffMatrix64Step)
class SupriseQmixPom128step(PayoffMatrix128Step, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PayoffMatrix128Step)
class SupriseQmixPom256step(PayoffMatrix256Step, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PayoffMatrix256Step)

class SupriseQmixPZPistonBall(PettingZooPistonBall, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PettingZooPistonBall)
class SupriseQmixPZCooperativePong(PettingZooCooperativePong, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PettingZooCooperativePong)
class SupriseQmixPZSimpleSpread(PettingZooSimpleSpread, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PettingZooSimpleSpread)
class SupriseQmixPZPursuit(PettingZooPursuit, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PettingZooPursuit)

class SupriseQmixPP4P(PressurePlateLinear4P, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PressurePlateLinear4P)
class SupriseQmixPP5P(PressurePlateLinear5P, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PressurePlateLinear5P)
class SupriseQmixPP6P(PressurePlateLinear6P, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, PressurePlateLinear6P)

class SupriseQmixRWTiny2Ag(RWarehouseTiny2Ag, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, RWarehouseTiny2Ag)
class SupriseQmixRWSmall4Ag(RWarehouseSmall4Ag, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, RWarehouseSmall4Ag)
class SupriseQmixRWHard6Ag(RWarehouseHard6Ag, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class SupriseQmixSc3M(SC2_3m, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_3m)

# StarCraft2: 3m
class SupriseQmixScSparse3M(SC2_sparse_3m, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class SupriseQmixSc27Mvs30M(SC2_27m_vs_30m, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class SupriseQmixSc3Svs5Z(SC2_3s_vs_5z, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class SupriseQmixSc3S5Z(SC2_3s5z, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class SupriseQmixSc2Cvs64ZG(SC2_2c_vs_64zg, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class SupriseQmixScCorridor(SC2_Corridor, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class SupriseQmixSc5Mvs6M(SC2_5m_vs_6m, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class SupriseQmixSc6Hvs8Z(SC2_6h_vs_8z, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class SupriseQmixScMMM2(SC2_MMM2, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class SupriseQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class SupriseQmixSc2Svs1SC(SC2_2s_vs_1sc, SURPRISEQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SURPRISEQMIX, SC2_2s_vs_1sc)

######################################################################################################


######################################################################################################
# VIINTRINSICQMIX

# Payoff-matrix Game: K-step
class VIIntrinsicQmixPom64step(PayoffMatrix64Step, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix64Step)
class VIIntrinsicQmixPom128step(PayoffMatrix128Step, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix128Step)
class VIIntrinsicQmixPom256step(PayoffMatrix256Step, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix256Step)

class VIIntrinsicQmixPZPistonBall(PettingZooPistonBall, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PettingZooPistonBall)
class VIIntrinsicQmixPZCooperativePong(PettingZooCooperativePong, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PettingZooCooperativePong)
class VIIntrinsicQmixPZSimpleSpread(PettingZooSimpleSpread, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PettingZooSimpleSpread)
class VIIntrinsicQmixPZPursuit(PettingZooPursuit, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PettingZooPursuit)

class VIIntrinsicQmixPP4P(PressurePlateLinear4P, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PressurePlateLinear4P)
class VIIntrinsicQmixPP5P(PressurePlateLinear5P, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PressurePlateLinear5P)
class VIIntrinsicQmixPP6P(PressurePlateLinear6P, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PressurePlateLinear6P)

class VIIntrinsicQmixRWTiny2Ag(RWarehouseTiny2Ag, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, RWarehouseTiny2Ag)
class VIIntrinsicQmixRWSmall4Ag(RWarehouseSmall4Ag, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, RWarehouseSmall4Ag)
class VIIntrinsicQmixRWHard6Ag(RWarehouseHard6Ag, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIIntrinsicQmixSc3M(SC2_3m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_3m)

# StarCraft2: 3m
class VIIntrinsicQmixScSparse3M(SC2_sparse_3m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class VIIntrinsicQmixSc27Mvs30M(SC2_27m_vs_30m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIIntrinsicQmixSc3Svs5Z(SC2_3s_vs_5z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIIntrinsicQmixSc3S5Z(SC2_3s5z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIIntrinsicQmixSc2Cvs64ZG(SC2_2c_vs_64zg, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIIntrinsicQmixScCorridor(SC2_Corridor, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIIntrinsicQmixSc5Mvs6M(SC2_5m_vs_6m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIIntrinsicQmixSc6Hvs8Z(SC2_6h_vs_8z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIIntrinsicQmixScMMM2(SC2_MMM2, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIIntrinsicQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIIntrinsicQmixSc2Svs1SC(SC2_2s_vs_1sc, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class VIIntrinsicQmixSc2S3Z(SC2_2s3z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class VIIntrinsicQmixSc1C3S5Z(SC2_1c3s5z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class VIIntrinsicQmixSc10Mvs11M(SC2_10m_vs_11m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class VIIntrinsicQmixScBANEvsBANE(SC2_bane_vs_bane, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_bane_vs_bane)

class VIIntrinsicQmixPom4stepSparse(PayoffMatrix4StepSparse, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix4StepSparse)
class VIIntrinsicQmixPom8stepSparse(PayoffMatrix8StepSparse, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix8StepSparse)
class VIIntrinsicQmixPom16stepSparse(PayoffMatrix16StepSparse, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix16StepSparse)
class VIIntrinsicQmixPom32stepSparse(PayoffMatrix32StepSparse, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, PayoffMatrix32StepSparse)

# StarCraft2: SC2_8m
class VIIntrinsicQmixSc8M(SC2_8m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_8m)

# StarCraft2: SC2_2m_vs_1z
class VIIntrinsicQmixSc2Mvs1Z(SC2_2m_vs_1z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_2m_vs_1z)

# StarCraft2: SC2_MMM
class VIIntrinsicQmixScMMM(SC2_MMM, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_MMM)

# StarCraft2: SC2_3s_vs_3z
class VIIntrinsicQmixSc3Svs3Z(SC2_3s_vs_3z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_3s_vs_3z)

# StarCraft2: SC2_3s_vs_4z
class VIIntrinsicQmixSc3Svs4Z(SC2_3s_vs_4z, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_3s_vs_4z)

# StarCraft2: SC2_so_many_baneling
class VIIntrinsicQmixScSoManyBaneling(SC2_so_many_baneling, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_so_many_baneling)

# StarCraft2: SC2_25m
class VIIntrinsicQmixSc25M(SC2_25m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_25m)

# StarCraft2: SC2_8m_vs_9m
class VIIntrinsicQmixSc8Mvs9M(SC2_8m_vs_9m, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2_8m_vs_9m)

# StarCraft2-v2: SC2V2
class VIIntrinsicQmixScv2_10GenZerg(SC2V2_10gen_zerg, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_10gen_zerg)

class VIIntrinsicQmixScv2_40GenZerg(SC2V2_40gen_zerg, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_40gen_zerg)

class VIIntrinsicQmixScv2_43GenZerg(SC2V2_43gen_zerg, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_43gen_zerg)

class VIIntrinsicQmixScv2_46GenZerg(SC2V2_46gen_zerg, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_46gen_zerg)

class VIIntrinsicQmixScv2_10GenProtoss(SC2V2_10gen_protoss, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_10gen_protoss)

class VIIntrinsicQmixScv2_40GenProtoss(SC2V2_40gen_protoss, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_40gen_protoss)

class VIIntrinsicQmixScv2_43GenProtoss(SC2V2_43gen_protoss, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_43gen_protoss)

class VIIntrinsicQmixScv2_46GenProtoss(SC2V2_46gen_protoss, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_46gen_protoss)

class VIIntrinsicQmixScv2_10GenTerran(SC2V2_10gen_terran, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_10gen_terran)

class VIIntrinsicQmixScv2_40GenTerran(SC2V2_40gen_terran, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_40gen_terran)

class VIIntrinsicQmixScv2_43GenTerran(SC2V2_43gen_terran, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_43gen_terran)

class VIIntrinsicQmixScv2_46GenTerran(SC2V2_46gen_terran, VIINTRINSICQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQMIX, SC2V2_46gen_terran)
######################################################################################################


######################################################################################################
# VIUTILEXPQMIX

# Payoff-matrix Game: K-step
class VIUtilExpQmixPom64step(PayoffMatrix64Step, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PayoffMatrix64Step)
class VIUtilExpQmixPom128step(PayoffMatrix128Step, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PayoffMatrix128Step)
class VIUtilExpQmixPom256step(PayoffMatrix256Step, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PayoffMatrix256Step)

class VIUtilExpQmixPZPistonBall(PettingZooPistonBall, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PettingZooPistonBall)
class VIUtilExpQmixPZCooperativePong(PettingZooCooperativePong, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PettingZooCooperativePong)
class VIUtilExpQmixPZSimpleSpread(PettingZooSimpleSpread, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PettingZooSimpleSpread)
class VIUtilExpQmixPZPursuit(PettingZooPursuit, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PettingZooPursuit)

class VIUtilExpQmixPP4P(PressurePlateLinear4P, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PressurePlateLinear4P)
class VIUtilExpQmixPP5P(PressurePlateLinear5P, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PressurePlateLinear5P)
class VIUtilExpQmixPP6P(PressurePlateLinear6P, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, PressurePlateLinear6P)

class VIUtilExpQmixRWTiny2Ag(RWarehouseTiny2Ag, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, RWarehouseTiny2Ag)
class VIUtilExpQmixRWSmall4Ag(RWarehouseSmall4Ag, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, RWarehouseSmall4Ag)
class VIUtilExpQmixRWHard6Ag(RWarehouseHard6Ag, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIUtilExpQmixSc3M(SC2_3m, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class VIUtilExpQmixSc27Mvs30M(SC2_27m_vs_30m, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIUtilExpQmixSc3Svs5Z(SC2_3s_vs_5z, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIUtilExpQmixSc3S5Z(SC2_3s5z, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIUtilExpQmixSc2Cvs64ZG(SC2_2c_vs_64zg, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIUtilExpQmixScCorridor(SC2_Corridor, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIUtilExpQmixSc5Mvs6M(SC2_5m_vs_6m, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIUtilExpQmixSc6Hvs8Z(SC2_6h_vs_8z, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIUtilExpQmixScMMM2(SC2_MMM2, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIUtilExpQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIUtilExpQmixSc2Svs1SC(SC2_2s_vs_1sc, VIUTILEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIUTILEXPQMIX, SC2_2s_vs_1sc)

######################################################################################################


######################################################################################################
# VIEXPQMIX

# Payoff-matrix Game: K-step
class VIExpQmixPom64step(PayoffMatrix64Step, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PayoffMatrix64Step)
class VIExpQmixPom128step(PayoffMatrix128Step, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PayoffMatrix128Step)
class VIExpQmixPom256step(PayoffMatrix256Step, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PayoffMatrix256Step)

class VIExpQmixPZPistonBall(PettingZooPistonBall, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PettingZooPistonBall)
class VIExpQmixPZCooperativePong(PettingZooCooperativePong, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PettingZooCooperativePong)
class VIExpQmixPZSimpleSpread(PettingZooSimpleSpread, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PettingZooSimpleSpread)
class VIExpQmixPZPursuit(PettingZooPursuit, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PettingZooPursuit)

class VIExpQmixPP4P(PressurePlateLinear4P, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PressurePlateLinear4P)
class VIExpQmixPP5P(PressurePlateLinear5P, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PressurePlateLinear5P)
class VIExpQmixPP6P(PressurePlateLinear6P, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, PressurePlateLinear6P)

class VIExpQmixRWTiny2Ag(RWarehouseTiny2Ag, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, RWarehouseTiny2Ag)
class VIExpQmixRWSmall4Ag(RWarehouseSmall4Ag, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, RWarehouseSmall4Ag)
class VIExpQmixRWHard6Ag(RWarehouseHard6Ag, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIExpQmixSc3M(SC2_3m, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class VIExpQmixSc27Mvs30M(SC2_27m_vs_30m, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIExpQmixSc3Svs5Z(SC2_3s_vs_5z, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIExpQmixSc3S5Z(SC2_3s5z, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIExpQmixSc2Cvs64ZG(SC2_2c_vs_64zg, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIExpQmixScCorridor(SC2_Corridor, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIExpQmixSc5Mvs6M(SC2_5m_vs_6m, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIExpQmixSc6Hvs8Z(SC2_6h_vs_8z, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIExpQmixScMMM2(SC2_MMM2, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIExpQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIExpQmixSc2Svs1SC(SC2_2s_vs_1sc, VIEXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIEXPQMIX, SC2_2s_vs_1sc)

######################################################################################################


######################################################################################################
# EXPQMIX

class ExpQmixPom64step(PayoffMatrix64Step, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix64Step)
class ExpQmixPom128step(PayoffMatrix128Step, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix128Step)
class ExpQmixPom256step(PayoffMatrix256Step, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix256Step)

class ExpQmixPZPistonBall(PettingZooPistonBall, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooPistonBall)
class ExpQmixPZCooperativePong(PettingZooCooperativePong, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooCooperativePong)
class ExpQmixPZSimpleSpread(PettingZooSimpleSpread, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooSimpleSpread)
class ExpQmixPZPursuit(PettingZooPursuit, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PettingZooPursuit)

class ExpQmixPP4P(PressurePlateLinear4P, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PressurePlateLinear4P)
class ExpQmixPP5P(PressurePlateLinear5P, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PressurePlateLinear5P)
class ExpQmixPP6P(PressurePlateLinear6P, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PressurePlateLinear6P)

class ExpQmixRWTiny2Ag(RWarehouseTiny2Ag, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, RWarehouseTiny2Ag)
class ExpQmixRWSmall4Ag(RWarehouseSmall4Ag, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, RWarehouseSmall4Ag)
class ExpQmixRWHard6Ag(RWarehouseHard6Ag, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, RWarehouseHard6Ag)

class ExpQmixSc3M(SC2_3m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3m)

class ExpQmixSc27Mvs30M(SC2_27m_vs_30m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_27m_vs_30m)

class ExpQmixSc3Svs5Z(SC2_3s_vs_5z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s_vs_5z)

class ExpQmixSc3S5Z(SC2_3s5z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s5z)

class ExpQmixSc2Cvs64ZG(SC2_2c_vs_64zg, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_2c_vs_64zg)

class ExpQmixScCorridor(SC2_Corridor, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_Corridor)

class ExpQmixSc5Mvs6M(SC2_5m_vs_6m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_5m_vs_6m)

class ExpQmixSc6Hvs8Z(SC2_6h_vs_8z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpQmixScMMM2(SC2_MMM2, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpQmixSc2Svs1SC(SC2_2s_vs_1sc, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class ExpQmixSc2S3Z(SC2_2s3z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class ExpQmixSc1C3S5Z(SC2_1c3s5z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class ExpQmixSc10Mvs11M(SC2_10m_vs_11m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class ExpQmixScBANEvsBANE(SC2_bane_vs_bane, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_bane_vs_bane)

class ExpQmixPom4stepSparse(PayoffMatrix4StepSparse, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix4StepSparse)
class ExpQmixPom8stepSparse(PayoffMatrix8StepSparse, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix8StepSparse)
class ExpQmixPom16stepSparse(PayoffMatrix16StepSparse, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix16StepSparse)
class ExpQmixPom32stepSparse(PayoffMatrix32StepSparse, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, PayoffMatrix32StepSparse)

# StarCraft2: SC2_8m
class ExpQmixSc8M(SC2_8m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_8m)

# StarCraft2: SC2_2m_vs_1z
class ExpQmixSc2Mvs1Z(SC2_2m_vs_1z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_2m_vs_1z)

# StarCraft2: SC2_MMM
class ExpQmixScMMM(SC2_MMM, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_MMM)

# StarCraft2: SC2_3s_vs_3z
class ExpQmixSc3Svs3Z(SC2_3s_vs_3z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s_vs_3z)

# StarCraft2: SC2_3s_vs_4z
class ExpQmixSc3Svs4Z(SC2_3s_vs_4z, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_3s_vs_4z)

# StarCraft2: SC2_so_many_baneling
class ExpQmixScSoManyBaneling(SC2_so_many_baneling, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_so_many_baneling)

# StarCraft2: SC2_25m
class ExpQmixSc25M(SC2_25m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_25m)

# StarCraft2: SC2_8m_vs_9m
class ExpQmixSc8Mvs9M(SC2_8m_vs_9m, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, SC2_8m_vs_9m)

# STAG HUNT
class ExpQmixStagHunt(STAGHUNT, EXPQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQMIX, STAGHUNT)
######################################################################################################


######################################################################################################
# EMC

# Payoff-matrix Game: K-step
class EmcPom64step(PayoffMatrix64Step, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix64Step)
class EmcPom128step(PayoffMatrix128Step, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix128Step)
class EmcPom256step(PayoffMatrix256Step, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix256Step)

class EmcPZPistonBall(PettingZooPistonBall, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PettingZooPistonBall)
class EmcPZCooperativePong(PettingZooCooperativePong, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PettingZooCooperativePong)
class EmcPZSimpleSpread(PettingZooSimpleSpread, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PettingZooSimpleSpread)
class EmcPZPursuit(PettingZooPursuit, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PettingZooPursuit)

class EmcPP4P(PressurePlateLinear4P, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PressurePlateLinear4P)
class EmcPP5P(PressurePlateLinear5P, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PressurePlateLinear5P)
class EmcPP6P(PressurePlateLinear6P, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PressurePlateLinear6P)

class EmcRWTiny2Ag(RWarehouseTiny2Ag, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, RWarehouseTiny2Ag)
class EmcRWSmall4Ag(RWarehouseSmall4Ag, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, RWarehouseSmall4Ag)
class EmcRWHard6Ag(RWarehouseHard6Ag, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, RWarehouseHard6Ag)

# StarCraft2: 3m
class EmcSc3M(SC2_3m, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_3m)

# StarCraft2: 3m
class EmcScSparse3M(SC2_sparse_3m, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class EmcSc27Mvs30M(SC2_27m_vs_30m, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class EmcSc3Svs5Z(SC2_3s_vs_5z, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class EmcSc3S5Z(SC2_3s5z, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class EmcSc2Cvs64ZG(SC2_2c_vs_64zg, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_2c_vs_64zg)

# StarCraft2: corridor
class EmcScCorridor(SC2_Corridor, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class EmcSc5Mvs6M(SC2_5m_vs_6m, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class EmcSc6Hvs8Z(SC2_6h_vs_8z, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_6h_vs_8z)

# StarCraft2: MMM2
class EmcScMMM2(SC2_MMM2, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class EmcSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class EmcSc2Svs1SC(SC2_2s_vs_1sc, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class EmcSc2S3Z(SC2_2s3z, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class EmcSc1C3S5Z(SC2_1c3s5z, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class EmcSc10Mvs11M(SC2_10m_vs_11m, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class EmcScBANEvsBANE(SC2_bane_vs_bane, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, SC2_bane_vs_bane)

class EmcPom4stepSparse(PayoffMatrix4StepSparse, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix4StepSparse)
class EmcPom8stepSparse(PayoffMatrix8StepSparse, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix8StepSparse)
class EmcPom16stepSparse(PayoffMatrix16StepSparse, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix16StepSparse)
class EmcPom32stepSparse(PayoffMatrix32StepSparse, EMC, General):
    param_overlapped_dict = check_parameters_overlapped(General, EMC, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# SEMIEMCQMIX

# Payoff-matrix Game: K-step
class SemiEmcQmixPom64step(PayoffMatrix64Step, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix64Step)
class SemiEmcQmixPom128step(PayoffMatrix128Step, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix128Step)
class SemiEmcQmixPom256step(PayoffMatrix256Step, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix256Step)

class SemiEmcQmixPZPistonBall(PettingZooPistonBall, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooPistonBall)
class SemiEmcQmixPZCooperativePong(PettingZooCooperativePong, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooCooperativePong)
class SemiEmcQmixPZSimpleSpread(PettingZooSimpleSpread, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooSimpleSpread)
class SemiEmcQmixPZPursuit(PettingZooPursuit, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PettingZooPursuit)

class SemiEmcQmixPP4P(PressurePlateLinear4P, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PressurePlateLinear4P)
class SemiEmcQmixPP5P(PressurePlateLinear5P, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PressurePlateLinear5P)
class SemiEmcQmixPP6P(PressurePlateLinear6P, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PressurePlateLinear6P)

class SemiEmcQmixRWTiny2Ag(RWarehouseTiny2Ag, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, RWarehouseTiny2Ag)
class SemiEmcQmixRWSmall4Ag(RWarehouseSmall4Ag, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, RWarehouseSmall4Ag)
class SemiEmcQmixRWHard6Ag(RWarehouseHard6Ag, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class SemiEmcQmixSc3M(SC2_3m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class SemiEmcQmixSc27Mvs30M(SC2_27m_vs_30m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class SemiEmcQmixSc3Svs5Z(SC2_3s_vs_5z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3s_vs_5z)

class SemiEmcQmixSc3S5Z(SC2_3s5z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3s5z)

class SemiEmcQmixSc2Cvs64ZG(SC2_2c_vs_64zg, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_2c_vs_64zg)

class SemiEmcQmixScCorridor(SC2_Corridor, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_Corridor)

class SemiEmcQmixSc5Mvs6M(SC2_5m_vs_6m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_5m_vs_6m)

class SemiEmcQmixSc6Hvs8Z(SC2_6h_vs_8z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class SemiEmcQmixScMMM2(SC2_MMM2, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class SemiEmcQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class SemiEmcQmixSc2Svs1SC(SC2_2s_vs_1sc, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class SemiEmcQmixSc2S3Z(SC2_2s3z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class SemiEmcQmixSc1C3S5Z(SC2_1c3s5z, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class SemiEmcQmixSc10Mvs11M(SC2_10m_vs_11m, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class SemiEmcQmixScBANEvsBANE(SC2_bane_vs_bane, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2_bane_vs_bane)

class SemiEmcQmixPom4stepSparse(PayoffMatrix4StepSparse, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix4StepSparse)
class SemiEmcQmixPom8stepSparse(PayoffMatrix8StepSparse, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix8StepSparse)
class SemiEmcQmixPom16stepSparse(PayoffMatrix16StepSparse, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix16StepSparse)
class SemiEmcQmixPom32stepSparse(PayoffMatrix32StepSparse, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, PayoffMatrix32StepSparse)

# StarCraft2-v2: SC2V2
class SemiEmcQmixScv2_10GenZerg(SC2V2_10gen_zerg, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_10gen_zerg)

class SemiEmcQmixScv2_40GenZerg(SC2V2_40gen_zerg, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_40gen_zerg)

class SemiEmcQmixScv2_43GenZerg(SC2V2_43gen_zerg, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_43gen_zerg)

class SemiEmcQmixScv2_46GenZerg(SC2V2_46gen_zerg, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_46gen_zerg)

class SemiEmcQmixScv2_10GenProtoss(SC2V2_10gen_protoss, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_10gen_protoss)

class SemiEmcQmixScv2_40GenProtoss(SC2V2_40gen_protoss, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_40gen_protoss)

class SemiEmcQmixScv2_43GenProtoss(SC2V2_43gen_protoss, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_43gen_protoss)

class SemiEmcQmixScv2_46GenProtoss(SC2V2_46gen_protoss, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_46gen_protoss)

class SemiEmcQmixScv2_10GenTerran(SC2V2_10gen_terran, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_10gen_terran)

class SemiEmcQmixScv2_40GenTerran(SC2V2_40gen_terran, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_40gen_terran)

class SemiEmcQmixScv2_43GenTerran(SC2V2_43gen_terran, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_43gen_terran)

class SemiEmcQmixScv2_46GenTerran(SC2V2_46gen_terran, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, SC2V2_46gen_terran)

# STAG HUNT
class SemiEmcQmixStagHunt(STAGHUNT, SEMIEMCQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, SEMIEMCQMIX, STAGHUNT)
######################################################################################################


######################################################################################################
# RNDQMIX

# Payoff-matrix Game: K-step
class RndQmixPom64step(PayoffMatrix64Step, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix64Step)
class RndQmixPom128step(PayoffMatrix128Step, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix128Step)
class RndQmixPom256step(PayoffMatrix256Step, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix256Step)

class RndQmixPZPistonBall(PettingZooPistonBall, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooPistonBall)
class RndQmixPZCooperativePong(PettingZooCooperativePong, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooCooperativePong)
class RndQmixPZSimpleSpread(PettingZooSimpleSpread, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooSimpleSpread)
class RndQmixPZPursuit(PettingZooPursuit, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PettingZooPursuit)

class RndQmixPP4P(PressurePlateLinear4P, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PressurePlateLinear4P)
class RndQmixPP5P(PressurePlateLinear5P, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PressurePlateLinear5P)
class RndQmixPP6P(PressurePlateLinear6P, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PressurePlateLinear6P)

class RndQmixRWTiny2Ag(RWarehouseTiny2Ag, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, RWarehouseTiny2Ag)
class RndQmixRWSmall4Ag(RWarehouseSmall4Ag, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, RWarehouseSmall4Ag)
class RndQmixRWHard6Ag(RWarehouseHard6Ag, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class RndQmixSc3M(SC2_3m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class RndQmixSc27Mvs30M(SC2_27m_vs_30m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class RndQmixSc3Svs5Z(SC2_3s_vs_5z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3s_vs_5z)

class RndQmixSc3S5Z(SC2_3s5z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3s5z)

class RndQmixSc2Cvs64ZG(SC2_2c_vs_64zg, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_2c_vs_64zg)

class RndQmixScCorridor(SC2_Corridor, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_Corridor)

class RndQmixSc5Mvs6M(SC2_5m_vs_6m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_5m_vs_6m)

class RndQmixSc6Hvs8Z(SC2_6h_vs_8z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class RndQmixScMMM2(SC2_MMM2, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class RndQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class RndQmixSc2Svs1SC(SC2_2s_vs_1sc, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class RndQmixSc2S3Z(SC2_2s3z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class RndQmixSc1C3S5Z(SC2_1c3s5z, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class RndQmixSc10Mvs11M(SC2_10m_vs_11m, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class RndQmixScBANEvsBANE(SC2_bane_vs_bane, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2_bane_vs_bane)

class RndQmixPom4stepSparse(PayoffMatrix4StepSparse, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix4StepSparse)
class RndQmixPom8stepSparse(PayoffMatrix8StepSparse, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix8StepSparse)
class RndQmixPom16stepSparse(PayoffMatrix16StepSparse, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix16StepSparse)
class RndQmixPom32stepSparse(PayoffMatrix32StepSparse, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, PayoffMatrix32StepSparse)

# StarCraft2-v2: SC2V2
class RndQmixScv2_10GenZerg(SC2V2_10gen_zerg, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_10gen_zerg)

class RndQmixScv2_40GenZerg(SC2V2_40gen_zerg, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_40gen_zerg)

class RndQmixScv2_43GenZerg(SC2V2_43gen_zerg, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_43gen_zerg)

class RndQmixScv2_46GenZerg(SC2V2_46gen_zerg, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_46gen_zerg)

class RndQmixScv2_10GenProtoss(SC2V2_10gen_protoss, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_10gen_protoss)

class RndQmixScv2_40GenProtoss(SC2V2_40gen_protoss, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_40gen_protoss)

class RndQmixScv2_43GenProtoss(SC2V2_43gen_protoss, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_43gen_protoss)

class RndQmixScv2_46GenProtoss(SC2V2_46gen_protoss, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_46gen_protoss)

class RndQmixScv2_10GenTerran(SC2V2_10gen_terran, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_10gen_terran)

class RndQmixScv2_40GenTerran(SC2V2_40gen_terran, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_40gen_terran)

class RndQmixScv2_43GenTerran(SC2V2_43gen_terran, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_43gen_terran)

class RndQmixScv2_46GenTerran(SC2V2_46gen_terran, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, SC2V2_46gen_terran)

# STAG HUNT
class RndQmixStagHunt(STAGHUNT, RNDQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, RNDQMIX, STAGHUNT)
######################################################################################################


######################################################################################################
# ICMQMIX

class IcmQmixPom64step(PayoffMatrix64Step, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix64Step)
class IcmQmixPom128step(PayoffMatrix128Step, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix128Step)
class IcmQmixPom256step(PayoffMatrix256Step, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix256Step)

class IcmQmixPZPistonBall(PettingZooPistonBall, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooPistonBall)
class IcmQmixPZCooperativePong(PettingZooCooperativePong, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooCooperativePong)
class IcmQmixPZSimpleSpread(PettingZooSimpleSpread, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooSimpleSpread)
class IcmQmixPZPursuit(PettingZooPursuit, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PettingZooPursuit)

class IcmQmixPP4P(PressurePlateLinear4P, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PressurePlateLinear4P)
class IcmQmixPP5P(PressurePlateLinear5P, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PressurePlateLinear5P)
class IcmQmixPP6P(PressurePlateLinear6P, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PressurePlateLinear6P)

class IcmQmixRWTiny2Ag(RWarehouseTiny2Ag, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, RWarehouseTiny2Ag)
class IcmQmixRWSmall4Ag(RWarehouseSmall4Ag, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, RWarehouseSmall4Ag)
class IcmQmixRWHard6Ag(RWarehouseHard6Ag, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, RWarehouseHard6Ag)

class IcmQmixSc3M(SC2_3m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3m)

class IcmQmixSc27Mvs30M(SC2_27m_vs_30m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_27m_vs_30m)

class IcmQmixSc3Svs5Z(SC2_3s_vs_5z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3s_vs_5z)

class IcmQmixSc3S5Z(SC2_3s5z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3s5z)

class IcmQmixSc2Cvs64ZG(SC2_2c_vs_64zg, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_2c_vs_64zg)

class IcmQmixScCorridor(SC2_Corridor, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_Corridor)

class IcmQmixSc5Mvs6M(SC2_5m_vs_6m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_5m_vs_6m)

class IcmQmixSc6Hvs8Z(SC2_6h_vs_8z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class IcmQmixScMMM2(SC2_MMM2, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class IcmQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class IcmQmixSc2Svs1SC(SC2_2s_vs_1sc, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class IcmQmixSc2S3Z(SC2_2s3z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class IcmQmixSc1C3S5Z(SC2_1c3s5z, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class IcmQmixSc10Mvs11M(SC2_10m_vs_11m, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class IcmQmixScBANEvsBANE(SC2_bane_vs_bane, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, SC2_bane_vs_bane)

class IcmQmixPom4stepSparse(PayoffMatrix4StepSparse, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix4StepSparse)
class IcmQmixPom8stepSparse(PayoffMatrix8StepSparse, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix8StepSparse)
class IcmQmixPom16stepSparse(PayoffMatrix16StepSparse, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix16StepSparse)
class IcmQmixPom32stepSparse(PayoffMatrix32StepSparse, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, PayoffMatrix32StepSparse)

# STAG HUNT
class IcmQmixStagHunt(STAGHUNT, ICMQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICMQMIX, STAGHUNT)
######################################################################################################


######################################################################################################
# QTRAN

class QtranPom64step(PayoffMatrix64Step, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix64Step)
class QtranPom128step(PayoffMatrix128Step, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix128Step)
class QtranPom256step(PayoffMatrix256Step, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix256Step)

class QtranPZPistonBall(PettingZooPistonBall, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooPistonBall)
class QtranPZCooperativePong(PettingZooCooperativePong, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooCooperativePong)
class QtranPZSimpleSpread(PettingZooSimpleSpread, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooSimpleSpread)
class QtranPZPursuit(PettingZooPursuit, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PettingZooPursuit)

class QtranPP4P(PressurePlateLinear4P, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PressurePlateLinear4P)
class QtranPP5P(PressurePlateLinear5P, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PressurePlateLinear5P)
class QtranPP6P(PressurePlateLinear6P, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PressurePlateLinear6P)

class QtranRWTiny2Ag(RWarehouseTiny2Ag, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, RWarehouseTiny2Ag)
class QtranRWSmall4Ag(RWarehouseSmall4Ag, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, RWarehouseSmall4Ag)
class QtranRWHard6Ag(RWarehouseHard6Ag, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, RWarehouseHard6Ag)

class QtranSc3M(SC2_3m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3m)

class QtranSc27Mvs30M(SC2_27m_vs_30m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_27m_vs_30m)

class QtranSc3Svs5Z(SC2_3s_vs_5z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3s_vs_5z)

class QtranSc3S5Z(SC2_3s5z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3s5z)

class QtranSc2Cvs64ZG(SC2_2c_vs_64zg, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_2c_vs_64zg)

class QtranScCorridor(SC2_Corridor, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_Corridor)

class QtranSc5Mvs6M(SC2_5m_vs_6m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_5m_vs_6m)

class QtranSc6Hvs8Z(SC2_6h_vs_8z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class QtranScMMM2(SC2_MMM2, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class QtranSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class QtranSc2Svs1SC(SC2_2s_vs_1sc, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class QtranSc2S3Z(SC2_2s3z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class QtranSc1C3S5Z(SC2_1c3s5z, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class QtranSc10Mvs11M(SC2_10m_vs_11m, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class QtranScBANEvsBANE(SC2_bane_vs_bane, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, SC2_bane_vs_bane)

class QtranPom4stepSparse(PayoffMatrix4StepSparse, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix4StepSparse)
class QtranPom8stepSparse(PayoffMatrix8StepSparse, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix8StepSparse)
class QtranPom16stepSparse(PayoffMatrix16StepSparse, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix16StepSparse)
class QtranPom32stepSparse(PayoffMatrix32StepSparse, QTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, QTRAN, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# VIIntrinsic Qtran

# Payoff-matrix Game: K-step
class VIIntrinsicQtranPom64step(PayoffMatrix64Step, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix64Step)
class VIIntrinsicQtranPom128step(PayoffMatrix128Step, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix128Step)
class VIIntrinsicQtranPom256step(PayoffMatrix256Step, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix256Step)
class VIIntrinsicQtranPom4stepSparse(PayoffMatrix4StepSparse, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix4StepSparse)
class VIIntrinsicQtranPom8stepSparse(PayoffMatrix8StepSparse, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix8StepSparse)
class VIIntrinsicQtranPom16stepSparse(PayoffMatrix16StepSparse, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix16StepSparse)
class VIIntrinsicQtranPom32stepSparse(PayoffMatrix32StepSparse, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PayoffMatrix32StepSparse)

class VIIntrinsicQtranPZPistonBall(PettingZooPistonBall, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PettingZooPistonBall)
class VIIntrinsicQtranPZCooperativePong(PettingZooCooperativePong, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PettingZooCooperativePong)
class VIIntrinsicQtranPZSimpleSpread(PettingZooSimpleSpread, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PettingZooSimpleSpread)
class VIIntrinsicQtranPZPursuit(PettingZooPursuit, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PettingZooPursuit)

class VIIntrinsicQtranPP4P(PressurePlateLinear4P, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PressurePlateLinear4P)
class VIIntrinsicQtranPP5P(PressurePlateLinear5P, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PressurePlateLinear5P)
class VIIntrinsicQtranPP6P(PressurePlateLinear6P, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, PressurePlateLinear6P)

class VIIntrinsicQtranRWTiny2Ag(RWarehouseTiny2Ag, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, RWarehouseTiny2Ag)
class VIIntrinsicQtranRWSmall4Ag(RWarehouseSmall4Ag, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, RWarehouseSmall4Ag)
class VIIntrinsicQtranRWHard6Ag(RWarehouseHard6Ag, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIIntrinsicQtranSc3M(SC2_3m, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_3m)

# StarCraft2: 3m
class VIIntrinsicQtranScSparse3M(SC2_sparse_3m, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class VIIntrinsicQtranSc27Mvs30M(SC2_27m_vs_30m, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIIntrinsicQtranSc3Svs5Z(SC2_3s_vs_5z, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIIntrinsicQtranSc3S5Z(SC2_3s5z, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIIntrinsicQtranSc2Cvs64ZG(SC2_2c_vs_64zg, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIIntrinsicQtranScCorridor(SC2_Corridor, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIIntrinsicQtranSc5Mvs6M(SC2_5m_vs_6m, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIIntrinsicQtranSc6Hvs8Z(SC2_6h_vs_8z, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIIntrinsicQtranScMMM2(SC2_MMM2, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIIntrinsicQtranSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIIntrinsicQtranSc2Svs1SC(SC2_2s_vs_1sc, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class VIIntrinsicQtranSc2S3Z(SC2_2s3z, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class VIIntrinsicQtranSc1C3S5Z(SC2_1c3s5z, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class VIIntrinsicQtranSc10Mvs11M(SC2_10m_vs_11m, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class VIIntrinsicQtranScBANEvsBANE(SC2_bane_vs_bane, VIINTRINSICQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQTRAN, SC2_bane_vs_bane)

######################################################################################################


######################################################################################################
# EXPQTRAN

class ExpQtranPom64step(PayoffMatrix64Step, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix64Step)
class ExpQtranPom128step(PayoffMatrix128Step, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix128Step)
class ExpQtranPom256step(PayoffMatrix256Step, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix256Step)

class ExpQtranPZPistonBall(PettingZooPistonBall, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooPistonBall)
class ExpQtranPZCooperativePong(PettingZooCooperativePong, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooCooperativePong)
class ExpQtranPZSimpleSpread(PettingZooSimpleSpread, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooSimpleSpread)
class ExpQtranPZPursuit(PettingZooPursuit, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PettingZooPursuit)

class ExpQtranPP4P(PressurePlateLinear4P, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PressurePlateLinear4P)
class ExpQtranPP5P(PressurePlateLinear5P, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PressurePlateLinear5P)
class ExpQtranPP6P(PressurePlateLinear6P, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PressurePlateLinear6P)

class ExpQtranRWTiny2Ag(RWarehouseTiny2Ag, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, RWarehouseTiny2Ag)
class ExpQtranRWSmall4Ag(RWarehouseSmall4Ag, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, RWarehouseSmall4Ag)
class ExpQtranRWHard6Ag(RWarehouseHard6Ag, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, RWarehouseHard6Ag)

class ExpQtranSc3M(SC2_3m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3m)


class ExpQtranSc27Mvs30M(SC2_27m_vs_30m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_27m_vs_30m)


class ExpQtranSc3Svs5Z(SC2_3s_vs_5z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3s_vs_5z)

class ExpQtranSc3S5Z(SC2_3s5z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3s5z)

class ExpQtranSc2Cvs64ZG(SC2_2c_vs_64zg, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_2c_vs_64zg)

class ExpQtranScCorridor(SC2_Corridor, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_Corridor)

class ExpQtranSc5Mvs6M(SC2_5m_vs_6m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_5m_vs_6m)

class ExpQtranSc6Hvs8Z(SC2_6h_vs_8z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpQtranScMMM2(SC2_MMM2, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpQtranSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpQtranSc2Svs1SC(SC2_2s_vs_1sc, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class ExpQtranSc2S3Z(SC2_2s3z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class ExpQtranSc1C3S5Z(SC2_1c3s5z, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class ExpQtranSc10Mvs11M(SC2_10m_vs_11m, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class ExpQtranScBANEvsBANE(SC2_bane_vs_bane, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, SC2_bane_vs_bane)

class ExpQtranPom4stepSparse(PayoffMatrix4StepSparse, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix4StepSparse)
class ExpQtranPom8stepSparse(PayoffMatrix8StepSparse, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix8StepSparse)
class ExpQtranPom16stepSparse(PayoffMatrix16StepSparse, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix16StepSparse)
class ExpQtranPom32stepSparse(PayoffMatrix32StepSparse, EXPQTRAN, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQTRAN, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# QPLEX

# Payoff-matrix Game: K-step
class QplexPom64step(PayoffMatrix64Step, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix64Step)
class QplexPom128step(PayoffMatrix128Step, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix128Step)
class QplexPom256step(PayoffMatrix256Step, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix256Step)

class QplexPZPistonBall(PettingZooPistonBall, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooPistonBall)
class QplexPZCooperativePong(PettingZooCooperativePong, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooCooperativePong)
class QplexPZSimpleSpread(PettingZooSimpleSpread, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooSimpleSpread)
class QplexPZPursuit(PettingZooPursuit, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PettingZooPursuit)

class QplexPP4P(PressurePlateLinear4P, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PressurePlateLinear4P)
class QplexPP5P(PressurePlateLinear5P, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PressurePlateLinear5P)
class QplexPP6P(PressurePlateLinear6P, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PressurePlateLinear6P)

class QplexRWTiny2Ag(RWarehouseTiny2Ag, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, RWarehouseTiny2Ag)
class QplexRWSmall4Ag(RWarehouseSmall4Ag, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, RWarehouseSmall4Ag)
class QplexRWHard6Ag(RWarehouseHard6Ag, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, RWarehouseHard6Ag)

# StarCraft2: 3m
class QplexSc3M(SC2_3m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3m)

# StarCraft2: 27m_vs_30m
class QplexSc27Mvs30M(SC2_27m_vs_30m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class QplexSc3Svs5Z(SC2_3s_vs_5z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3s_vs_5z)

class QplexSc3S5Z(SC2_3s5z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3s5z)

class QplexSc2Cvs64ZG(SC2_2c_vs_64zg, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_2c_vs_64zg)

class QplexScCorridor(SC2_Corridor, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_Corridor)

class QplexSc5Mvs6M(SC2_5m_vs_6m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_5m_vs_6m)

class QplexSc6Hvs8Z(SC2_6h_vs_8z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class QplexScMMM2(SC2_MMM2, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class QplexSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class QplexSc2Svs1SC(SC2_2s_vs_1sc, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class QplexSc2S3Z(SC2_2s3z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class QplexSc1C3S5Z(SC2_1c3s5z, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class QplexSc10Mvs11M(SC2_10m_vs_11m, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class QplexScBANEvsBANE(SC2_bane_vs_bane, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, SC2_bane_vs_bane)

class QplexPom4stepSparse(PayoffMatrix4StepSparse, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix4StepSparse)
class QplexPom8stepSparse(PayoffMatrix8StepSparse, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix8StepSparse)
class QplexPom16stepSparse(PayoffMatrix16StepSparse, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix16StepSparse)
class QplexPom32stepSparse(PayoffMatrix32StepSparse, QPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, QPLEX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# EXPQPLEX

# Payoff-matrix Game: K-step
class ExpQplexPom64step(PayoffMatrix64Step, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix64Step)
class ExpQplexPom128step(PayoffMatrix128Step, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix128Step)
class ExpQplexPom256step(PayoffMatrix256Step, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix256Step)

class ExpQplexPZPistonBall(PettingZooPistonBall, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooPistonBall)
class ExpQplexPZCooperativePong(PettingZooCooperativePong, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooCooperativePong)
class ExpQplexPZSimpleSpread(PettingZooSimpleSpread, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooSimpleSpread)
class ExpQplexPZPursuit(PettingZooPursuit, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PettingZooPursuit)

class ExpQplexPP4P(PressurePlateLinear4P, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PressurePlateLinear4P)
class ExpQplexPP5P(PressurePlateLinear5P, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PressurePlateLinear5P)
class ExpQplexPP6P(PressurePlateLinear6P, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PressurePlateLinear6P)

class ExpQplexRWTiny2Ag(RWarehouseTiny2Ag, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, RWarehouseTiny2Ag)
class ExpQplexRWSmall4Ag(RWarehouseSmall4Ag, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, RWarehouseSmall4Ag)
class ExpQplexRWHard6Ag(RWarehouseHard6Ag, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, RWarehouseHard6Ag)

# StarCraft2: 3m
class ExpQplexSc3M(SC2_3m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3m)


# StarCraft2: 27m_vs_30m
class ExpQplexSc27Mvs30M(SC2_27m_vs_30m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class ExpQplexSc3Svs5Z(SC2_3s_vs_5z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3s_vs_5z)

class ExpQplexSc3S5Z(SC2_3s5z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3s5z)

class ExpQplexSc2Cvs64ZG(SC2_2c_vs_64zg, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_2c_vs_64zg)

class ExpQplexScCorridor(SC2_Corridor, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_Corridor)

class ExpQplexSc5Mvs6M(SC2_5m_vs_6m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_5m_vs_6m)

class ExpQplexSc6Hvs8Z(SC2_6h_vs_8z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpQplexScMMM2(SC2_MMM2, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpQplexSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpQplexSc2Svs1SC(SC2_2s_vs_1sc, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class ExpQplexSc2S3Z(SC2_2s3z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class ExpQplexSc1C3S5Z(SC2_1c3s5z, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class ExpQplexSc10Mvs11M(SC2_10m_vs_11m, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class ExpQplexScBANEvsBANE(SC2_bane_vs_bane, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, SC2_bane_vs_bane)

class ExpQplexPom4stepSparse(PayoffMatrix4StepSparse, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix4StepSparse)
class ExpQplexPom8stepSparse(PayoffMatrix8StepSparse, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix8StepSparse)
class ExpQplexPom16stepSparse(PayoffMatrix16StepSparse, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix16StepSparse)
class ExpQplexPom32stepSparse(PayoffMatrix32StepSparse, EXPQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPQPLEX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# INTRINSIC QPLEX

# Payoff-matrix Game: K-step
class VIIntrinsicQplexPom64step(PayoffMatrix64Step, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix64Step)
class VIIntrinsicQplexPom128step(PayoffMatrix128Step, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix128Step)
class VIIntrinsicQplexPom256step(PayoffMatrix256Step, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix256Step)

class VIIntrinsicQplexPZPistonBall(PettingZooPistonBall, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PettingZooPistonBall)
class VIIntrinsicQplexPZCooperativePong(PettingZooCooperativePong, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PettingZooCooperativePong)
class VIIntrinsicQplexPZSimpleSpread(PettingZooSimpleSpread, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PettingZooSimpleSpread)
class VIIntrinsicQplexPZPursuit(PettingZooPursuit, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PettingZooPursuit)

class VIIntrinsicQplexPP4P(PressurePlateLinear4P, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PressurePlateLinear4P)
class VIIntrinsicQplexPP5P(PressurePlateLinear5P, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PressurePlateLinear5P)
class VIIntrinsicQplexPP6P(PressurePlateLinear6P, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PressurePlateLinear6P)

class VIIntrinsicQplexRWTiny2Ag(RWarehouseTiny2Ag, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, RWarehouseTiny2Ag)
class VIIntrinsicQplexRWSmall4Ag(RWarehouseSmall4Ag, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, RWarehouseSmall4Ag)
class VIIntrinsicQplexRWHard6Ag(RWarehouseHard6Ag, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIIntrinsicQplexSc3M(SC2_3m, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_3m)

# StarCraft2: 3m
class VIIntrinsicQplexScSparse3M(SC2_sparse_3m, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class VIIntrinsicQplexSc27Mvs30M(SC2_27m_vs_30m, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIIntrinsicQplexSc3Svs5Z(SC2_3s_vs_5z, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIIntrinsicQplexSc3S5Z(SC2_3s5z, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIIntrinsicQplexSc2Cvs64ZG(SC2_2c_vs_64zg, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIIntrinsicQplexScCorridor(SC2_Corridor, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIIntrinsicQplexSc5Mvs6M(SC2_5m_vs_6m, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIIntrinsicQplexSc6Hvs8Z(SC2_6h_vs_8z, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIIntrinsicQplexScMMM2(SC2_MMM2, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIIntrinsicQplexSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIIntrinsicQplexSc2Svs1SC(SC2_2s_vs_1sc, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class VIIntrinsicQplexSc2S3Z(SC2_2s3z, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class VIIntrinsicQplexSc1C3S5Z(SC2_1c3s5z, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class VIIntrinsicQplexSc10Mvs11M(SC2_10m_vs_11m, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class VIIntrinsicQplexScBANEvsBANE(SC2_bane_vs_bane, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, SC2_bane_vs_bane)

class VIIntrinsicQplexPom4stepSparse(PayoffMatrix4StepSparse, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix4StepSparse)
class VIIntrinsicQplexPom8stepSparse(PayoffMatrix8StepSparse, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix8StepSparse)
class VIIntrinsicQplexPom16stepSparse(PayoffMatrix16StepSparse, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix16StepSparse)
class VIIntrinsicQplexPom32stepSparse(PayoffMatrix32StepSparse, VIINTRINSICQPLEX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICQPLEX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# CWQMIX

# Payoff-matrix Game: K-step
class CwQmixPom64step(PayoffMatrix64Step, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix64Step)
class CwQmixPom128step(PayoffMatrix128Step, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix128Step)
class CwQmixPom256step(PayoffMatrix256Step, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix256Step)

class CwQmixPZPistonBall(PettingZooPistonBall, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooPistonBall)
class CwQmixPZCooperativePong(PettingZooCooperativePong, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooCooperativePong)
class CwQmixPZSimpleSpread(PettingZooSimpleSpread, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooSimpleSpread)
class CwQmixPZPursuit(PettingZooPursuit, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PettingZooPursuit)

class CwQmixPP4P(PressurePlateLinear4P, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PressurePlateLinear4P)
class CwQmixPP5P(PressurePlateLinear5P, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PressurePlateLinear5P)
class CwQmixPP6P(PressurePlateLinear6P, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PressurePlateLinear6P)

class CwQmixRWTiny2Ag(RWarehouseTiny2Ag, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, RWarehouseTiny2Ag)
class CwQmixRWSmall4Ag(RWarehouseSmall4Ag, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, RWarehouseSmall4Ag)
class CwQmixRWHard6Ag(RWarehouseHard6Ag, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class CwQmixSc3M(SC2_3m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class CwQmixSc27Mvs30M(SC2_27m_vs_30m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class CwQmixSc3Svs5Z(SC2_3s_vs_5z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3s_vs_5z)

class CwQmixSc3S5Z(SC2_3s5z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3s5z)

class CwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_2c_vs_64zg)

class CwQmixScCorridor(SC2_Corridor, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_Corridor)

class CwQmixSc5Mvs6M(SC2_5m_vs_6m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_5m_vs_6m)

class CwQmixSc6Hvs8Z(SC2_6h_vs_8z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class CwQmixScMMM2(SC2_MMM2, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class CwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class CwQmixSc2Svs1SC(SC2_2s_vs_1sc, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class CwQmixSc2S3Z(SC2_2s3z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class CwQmixSc1C3S5Z(SC2_1c3s5z, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class CwQmixSc10Mvs11M(SC2_10m_vs_11m, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class CwQmixScBANEvsBANE(SC2_bane_vs_bane, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, SC2_bane_vs_bane)

class CwQmixPom4stepSparse(PayoffMatrix4StepSparse, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix4StepSparse)
class CwQmixPom8stepSparse(PayoffMatrix8StepSparse, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix8StepSparse)
class CwQmixPom16stepSparse(PayoffMatrix16StepSparse, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix16StepSparse)
class CwQmixPom32stepSparse(PayoffMatrix32StepSparse, CWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, CWQMIX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# EXPCWQMIX

# Payoff-matrix Game: K-step
class ExpCwQmixPom64step(PayoffMatrix64Step, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix64Step)
class ExpCwQmixPom128step(PayoffMatrix128Step, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix128Step)
class ExpCwQmixPom256step(PayoffMatrix256Step, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix256Step)

class ExpCwQmixPZPistonBall(PettingZooPistonBall, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooPistonBall)
class ExpCwQmixPZCooperativePong(PettingZooCooperativePong, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooCooperativePong)
class ExpCwQmixPZSimpleSpread(PettingZooSimpleSpread, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooSimpleSpread)
class ExpCwQmixPZPursuit(PettingZooPursuit, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PettingZooPursuit)

class ExpCwQmixPP4P(PressurePlateLinear4P, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PressurePlateLinear4P)
class ExpCwQmixPP5P(PressurePlateLinear5P, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PressurePlateLinear5P)
class ExpCwQmixPP6P(PressurePlateLinear6P, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PressurePlateLinear6P)

class ExpCwQmixRWTiny2Ag(RWarehouseTiny2Ag, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, RWarehouseTiny2Ag)
class ExpCwQmixRWSmall4Ag(RWarehouseSmall4Ag, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, RWarehouseSmall4Ag)
class ExpCwQmixRWHard6Ag(RWarehouseHard6Ag, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class ExpCwQmixSc3M(SC2_3m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class ExpCwQmixSc27Mvs30M(SC2_27m_vs_30m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class ExpCwQmixSc3Svs5Z(SC2_3s_vs_5z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3s_vs_5z)

class ExpCwQmixSc3S5Z(SC2_3s5z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3s5z)

class ExpCwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_2c_vs_64zg)

class ExpCwQmixScCorridor(SC2_Corridor, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_Corridor)

class ExpCwQmixSc5Mvs6M(SC2_5m_vs_6m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_5m_vs_6m)

class ExpCwQmixSc6Hvs8Z(SC2_6h_vs_8z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpCwQmixScMMM2(SC2_MMM2, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpCwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpCwQmixSc2Svs1SC(SC2_2s_vs_1sc, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class ExpCwQmixSc2S3Z(SC2_2s3z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class ExpCwQmixSc1C3S5Z(SC2_1c3s5z, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class ExpCwQmixSc10Mvs11M(SC2_10m_vs_11m, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class ExpCwQmixScBANEvsBANE(SC2_bane_vs_bane, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, SC2_bane_vs_bane)

class ExpCwQmixPom4stepSparse(PayoffMatrix4StepSparse, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix4StepSparse)
class ExpCwQmixPom8stepSparse(PayoffMatrix8StepSparse, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix8StepSparse)
class ExpCwQmixPom16stepSparse(PayoffMatrix16StepSparse, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix16StepSparse)
class ExpCwQmixPom32stepSparse(PayoffMatrix32StepSparse, EXPCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPCWQMIX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# VIIntrinsic Cw-Qmix

# Payoff-matrix Game: K-step
class VIIntrinsicCwQmixPom64step(PayoffMatrix64Step, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix64Step)
class VIIntrinsicCwQmixPom128step(PayoffMatrix128Step, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix128Step)
class VIIntrinsicCwQmixPom256step(PayoffMatrix256Step, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix256Step)
class VIIntrinsicCwQmixPom4stepSparse(PayoffMatrix4StepSparse, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix4StepSparse)
class VIIntrinsicCwQmixPom8stepSparse(PayoffMatrix8StepSparse, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix8StepSparse)
class VIIntrinsicCwQmixPom16stepSparse(PayoffMatrix16StepSparse, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix16StepSparse)
class VIIntrinsicCwQmixPom32stepSparse(PayoffMatrix32StepSparse, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PayoffMatrix32StepSparse)

class VIIntrinsicCwQmixPZPistonBall(PettingZooPistonBall, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PettingZooPistonBall)
class VIIntrinsicCwQmixPZCooperativePong(PettingZooCooperativePong, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PettingZooCooperativePong)
class VIIntrinsicCwQmixPZSimpleSpread(PettingZooSimpleSpread, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PettingZooSimpleSpread)
class VIIntrinsicCwQmixPZPursuit(PettingZooPursuit, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PettingZooPursuit)

class VIIntrinsicCwQmixPP4P(PressurePlateLinear4P, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PressurePlateLinear4P)
class VIIntrinsicCwQmixPP5P(PressurePlateLinear5P, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PressurePlateLinear5P)
class VIIntrinsicCwQmixPP6P(PressurePlateLinear6P, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, PressurePlateLinear6P)

class VIIntrinsicCwQmixRWTiny2Ag(RWarehouseTiny2Ag, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, RWarehouseTiny2Ag)
class VIIntrinsicCwQmixRWSmall4Ag(RWarehouseSmall4Ag, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, RWarehouseSmall4Ag)
class VIIntrinsicCwQmixRWHard6Ag(RWarehouseHard6Ag, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIIntrinsicCwQmixSc3M(SC2_3m, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_3m)

# StarCraft2: 3m
class VIIntrinsicCwQmixScSparse3M(SC2_sparse_3m, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class VIIntrinsicCwQmixSc27Mvs30M(SC2_27m_vs_30m, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIIntrinsicCwQmixSc3Svs5Z(SC2_3s_vs_5z, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIIntrinsicCwQmixSc3S5Z(SC2_3s5z, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIIntrinsicCwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIIntrinsicCwQmixScCorridor(SC2_Corridor, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIIntrinsicCwQmixSc5Mvs6M(SC2_5m_vs_6m, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIIntrinsicCwQmixSc6Hvs8Z(SC2_6h_vs_8z, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIIntrinsicCwQmixScMMM2(SC2_MMM2, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIIntrinsicCwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIIntrinsicCwQmixSc2Svs1SC(SC2_2s_vs_1sc, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class VIIntrinsicCwQmixSc2S3Z(SC2_2s3z, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class VIIntrinsicCwQmixSc1C3S5Z(SC2_1c3s5z, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class VIIntrinsicCwQmixSc10Mvs11M(SC2_10m_vs_11m, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class VIIntrinsicCwQmixScBANEvsBANE(SC2_bane_vs_bane, VIINTRINSICCWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICCWQMIX, SC2_bane_vs_bane)

######################################################################################################


######################################################################################################
# OWQMIX

# Payoff-matrix Game: K-step
class OwQmixPom64step(PayoffMatrix64Step, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix64Step)
class OwQmixPom128step(PayoffMatrix128Step, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix128Step)
class OwQmixPom256step(PayoffMatrix256Step, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix256Step)

class OwQmixPZPistonBall(PettingZooPistonBall, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooPistonBall)
class OwQmixPZCooperativePong(PettingZooCooperativePong, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooCooperativePong)
class OwQmixPZSimpleSpread(PettingZooSimpleSpread, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooSimpleSpread)
class OwQmixPZPursuit(PettingZooPursuit, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PettingZooPursuit)

class OwQmixPP4P(PressurePlateLinear4P, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PressurePlateLinear4P)
class OwQmixPP5P(PressurePlateLinear5P, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PressurePlateLinear5P)
class OwQmixPP6P(PressurePlateLinear6P, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PressurePlateLinear6P)

class OwQmixRWTiny2Ag(RWarehouseTiny2Ag, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, RWarehouseTiny2Ag)
class OwQmixRWSmall4Ag(RWarehouseSmall4Ag, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, RWarehouseSmall4Ag)
class OwQmixRWHard6Ag(RWarehouseHard6Ag, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class OwQmixSc3M(SC2_3m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class OwQmixSc27Mvs30M(SC2_27m_vs_30m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class OwQmixSc3Svs5Z(SC2_3s_vs_5z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3s_vs_5z)

class OwQmixSc3S5Z(SC2_3s5z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3s5z)

class OwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_2c_vs_64zg)

class OwQmixScCorridor(SC2_Corridor, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_Corridor)

class OwQmixSc5Mvs6M(SC2_5m_vs_6m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_5m_vs_6m)

class OwQmixSc6Hvs8Z(SC2_6h_vs_8z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class OwQmixScMMM2(SC2_MMM2, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class OwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class OwQmixSc2Svs1SC(SC2_2s_vs_1sc, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class OwQmixSc2S3Z(SC2_2s3z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class OwQmixSc1C3S5Z(SC2_1c3s5z, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class OwQmixSc10Mvs11M(SC2_10m_vs_11m, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class OwQmixScBANEvsBANE(SC2_bane_vs_bane, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, SC2_bane_vs_bane)

class OwQmixPom4stepSparse(PayoffMatrix4StepSparse, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix4StepSparse)
class OwQmixPom8stepSparse(PayoffMatrix8StepSparse, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix8StepSparse)
class OwQmixPom16stepSparse(PayoffMatrix16StepSparse, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix16StepSparse)
class OwQmixPom32stepSparse(PayoffMatrix32StepSparse, OWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, OWQMIX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# VIIntrinsic Ow-Qmix

# Payoff-matrix Game: K-step
class VIIntrinsicOwQmixPom64step(PayoffMatrix64Step, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix64Step)
class VIIntrinsicOwQmixPom128step(PayoffMatrix128Step, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix128Step)
class VIIntrinsicOwQmixPom256step(PayoffMatrix256Step, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix256Step)
class VIIntrinsicOwQmixPom4stepSparse(PayoffMatrix4StepSparse, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix4StepSparse)
class VIIntrinsicOwQmixPom8stepSparse(PayoffMatrix8StepSparse, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix8StepSparse)
class VIIntrinsicOwQmixPom16stepSparse(PayoffMatrix16StepSparse, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix16StepSparse)
class VIIntrinsicOwQmixPom32stepSparse(PayoffMatrix32StepSparse, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PayoffMatrix32StepSparse)

class VIIntrinsicOwQmixPZPistonBall(PettingZooPistonBall, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PettingZooPistonBall)
class VIIntrinsicOwQmixPZCooperativePong(PettingZooCooperativePong, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PettingZooCooperativePong)
class VIIntrinsicOwQmixPZSimpleSpread(PettingZooSimpleSpread, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PettingZooSimpleSpread)
class VIIntrinsicOwQmixPZPursuit(PettingZooPursuit, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PettingZooPursuit)

class VIIntrinsicOwQmixPP4P(PressurePlateLinear4P, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PressurePlateLinear4P)
class VIIntrinsicOwQmixPP5P(PressurePlateLinear5P, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PressurePlateLinear5P)
class VIIntrinsicOwQmixPP6P(PressurePlateLinear6P, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, PressurePlateLinear6P)

class VIIntrinsicOwQmixRWTiny2Ag(RWarehouseTiny2Ag, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, RWarehouseTiny2Ag)
class VIIntrinsicOwQmixRWSmall4Ag(RWarehouseSmall4Ag, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, RWarehouseSmall4Ag)
class VIIntrinsicOwQmixRWHard6Ag(RWarehouseHard6Ag, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class VIIntrinsicOwQmixSc3M(SC2_3m, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_3m)

# StarCraft2: 3m
class VIIntrinsicOwQmixScSparse3M(SC2_sparse_3m, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_sparse_3m)

# StarCraft2: 27m_vs_30m
class VIIntrinsicOwQmixSc27Mvs30M(SC2_27m_vs_30m, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class VIIntrinsicOwQmixSc3Svs5Z(SC2_3s_vs_5z, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class VIIntrinsicOwQmixSc3S5Z(SC2_3s5z, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class VIIntrinsicOwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_2c_vs_64zg)

# StarCraft2: corridor
class VIIntrinsicOwQmixScCorridor(SC2_Corridor, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class VIIntrinsicOwQmixSc5Mvs6M(SC2_5m_vs_6m, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class VIIntrinsicOwQmixSc6Hvs8Z(SC2_6h_vs_8z, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class VIIntrinsicOwQmixScMMM2(SC2_MMM2, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class VIIntrinsicOwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class VIIntrinsicOwQmixSc2Svs1SC(SC2_2s_vs_1sc, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class VIIntrinsicOwQmixSc2S3Z(SC2_2s3z, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class VIIntrinsicOwQmixSc1C3S5Z(SC2_1c3s5z, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class VIIntrinsicOwQmixSc10Mvs11M(SC2_10m_vs_11m, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class VIIntrinsicOwQmixScBANEvsBANE(SC2_bane_vs_bane, VIINTRINSICOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, VIINTRINSICOWQMIX, SC2_bane_vs_bane)

######################################################################################################


######################################################################################################
# EXPOWQMIX

# Payoff-matrix Game: K-step
class ExpOwQmixPom64step(PayoffMatrix64Step, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix64Step)
class ExpOwQmixPom128step(PayoffMatrix128Step, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix128Step)
class ExpOwQmixPom256step(PayoffMatrix256Step, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix256Step)

class ExpOwQmixPZPistonBall(PettingZooPistonBall, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooPistonBall)
class ExpOwQmixPZCooperativePong(PettingZooCooperativePong, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooCooperativePong)
class ExpOwQmixPZSimpleSpread(PettingZooSimpleSpread, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooSimpleSpread)
class ExpOwQmixPZPursuit(PettingZooPursuit, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PettingZooPursuit)

class ExpOwQmixPP4P(PressurePlateLinear4P, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PressurePlateLinear4P)
class ExpOwQmixPP5P(PressurePlateLinear5P, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PressurePlateLinear5P)
class ExpOwQmixPP6P(PressurePlateLinear6P, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PressurePlateLinear6P)

class ExpOwQmixRWTiny2Ag(RWarehouseTiny2Ag, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, RWarehouseTiny2Ag)
class ExpOwQmixRWSmall4Ag(RWarehouseSmall4Ag, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, RWarehouseSmall4Ag)
class ExpOwQmixRWHard6Ag(RWarehouseHard6Ag, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, RWarehouseHard6Ag)

# StarCraft2: 3m
class ExpOwQmixSc3M(SC2_3m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3m)

# StarCraft2: 27m_vs_30m
class ExpOwQmixSc27Mvs30M(SC2_27m_vs_30m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class ExpOwQmixSc3Svs5Z(SC2_3s_vs_5z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3s_vs_5z)

class ExpOwQmixSc3S5Z(SC2_3s5z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3s5z)

class ExpOwQmixSc2Cvs64ZG(SC2_2c_vs_64zg, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_2c_vs_64zg)

class ExpOwQmixScCorridor(SC2_Corridor, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_Corridor)

class ExpOwQmixSc5Mvs6M(SC2_5m_vs_6m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_5m_vs_6m)

class ExpOwQmixSc6Hvs8Z(SC2_6h_vs_8z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ExpOwQmixScMMM2(SC2_MMM2, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ExpOwQmixSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class ExpOwQmixSc2Svs1SC(SC2_2s_vs_1sc, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class ExpOwQmixSc2S3Z(SC2_2s3z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class ExpOwQmixSc1C3S5Z(SC2_1c3s5z, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class ExpOwQmixSc10Mvs11M(SC2_10m_vs_11m, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class ExpOwQmixScBANEvsBANE(SC2_bane_vs_bane, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, SC2_bane_vs_bane)

class ExpOwQmixPom4stepSparse(PayoffMatrix4StepSparse, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix4StepSparse)
class ExpOwQmixPom8stepSparse(PayoffMatrix8StepSparse, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix8StepSparse)
class ExpOwQmixPom16stepSparse(PayoffMatrix16StepSparse, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix16StepSparse)
class ExpOwQmixPom32stepSparse(PayoffMatrix32StepSparse, EXPOWQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPOWQMIX, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# MAVEN

# Payoff-matrix Game: K-step
class MavenPom64step(PayoffMatrix64Step, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix64Step)
class MavenPom128step(PayoffMatrix128Step, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix128Step)
class MavenPom256step(PayoffMatrix256Step, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix256Step)

class MavenPZPistonBall(PettingZooPistonBall, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooPistonBall)
class MavenPZCooperativePong(PettingZooCooperativePong, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooCooperativePong)
class MavenPZSimpleSpread(PettingZooSimpleSpread, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooSimpleSpread)
class MavenPZPursuit(PettingZooPursuit, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PettingZooPursuit)

class MavenPP4P(PressurePlateLinear4P, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PressurePlateLinear4P)
class MavenPP5P(PressurePlateLinear5P, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PressurePlateLinear5P)
class MavenPP6P(PressurePlateLinear6P, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PressurePlateLinear6P)

class MavenRWTiny2Ag(RWarehouseTiny2Ag, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, RWarehouseTiny2Ag)
class MavenRWSmall4Ag(RWarehouseSmall4Ag, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, RWarehouseSmall4Ag)
class MavenRWHard6Ag(RWarehouseHard6Ag, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, RWarehouseHard6Ag)

# StarCraft2: 3m
class MavenSc3M(SC2_3m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3m)

# StarCraft2: 27m_vs_30m
class MavenSc27Mvs30M(SC2_27m_vs_30m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class MavenSc3Svs5Z(SC2_3s_vs_5z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class MavenSc3S5Z(SC2_3s5z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class MavenSc2Cvs64ZG(SC2_2c_vs_64zg, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_2c_vs_64zg)

# StarCraft2: corridor
class MavenScCorridor(SC2_Corridor, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class MavenSc5Mvs6M(SC2_5m_vs_6m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class MavenSc6Hvs8Z(SC2_6h_vs_8z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_6h_vs_8z)

# StarCraft2: MMM2
class MavenScMMM2(SC2_MMM2, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class MavenSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class MavenSc2Svs1SC(SC2_2s_vs_1sc, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class MavenSc2S3Z(SC2_2s3z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class MavenSc1C3S5Z(SC2_1c3s5z, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class MavenSc10Mvs11M(SC2_10m_vs_11m, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class MavenScBANEvsBANE(SC2_bane_vs_bane, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2_bane_vs_bane)

class MavenPom4stepSparse(PayoffMatrix4StepSparse, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix4StepSparse)
class MavenPom8stepSparse(PayoffMatrix8StepSparse, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix8StepSparse)
class MavenPom16stepSparse(PayoffMatrix16StepSparse, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix16StepSparse)
class MavenPom32stepSparse(PayoffMatrix32StepSparse, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, PayoffMatrix32StepSparse)

# StarCraft2-v2: SC2V2
class MavenScv2_10GenZerg(SC2V2_10gen_zerg, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_10gen_zerg)

class MavenScv2_40GenZerg(SC2V2_40gen_zerg, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_40gen_zerg)

class MavenScv2_43GenZerg(SC2V2_43gen_zerg, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_43gen_zerg)

class MavenScv2_46GenZerg(SC2V2_46gen_zerg, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_46gen_zerg)

class MavenScv2_10GenProtoss(SC2V2_10gen_protoss, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_10gen_protoss)

class MavenScv2_40GenProtoss(SC2V2_40gen_protoss, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_40gen_protoss)

class MavenScv2_43GenProtoss(SC2V2_43gen_protoss, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_43gen_protoss)

class MavenScv2_46GenProtoss(SC2V2_46gen_protoss, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_46gen_protoss)

class MavenScv2_10GenTerran(SC2V2_10gen_terran, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_10gen_terran)

class MavenScv2_40GenTerran(SC2V2_40gen_terran, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_40gen_terran)

class MavenScv2_43GenTerran(SC2V2_43gen_terran, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_43gen_terran)

class MavenScv2_46GenTerran(SC2V2_46gen_terran, MAVEN, General):
    param_overlapped_dict = check_parameters_overlapped(General, MAVEN, SC2V2_46gen_terran)
######################################################################################################


######################################################################################################
# LIIR

# Payoff-matrix Game: K-step
class LiirPom64step(PayoffMatrix64Step, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix64Step)
class LiirPom128step(PayoffMatrix128Step, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix128Step)
class LiirPom256step(PayoffMatrix256Step, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix256Step)

class LiirPZPistonBall(PettingZooPistonBall, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PettingZooPistonBall)
class LiirPZCooperativePong(PettingZooCooperativePong, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PettingZooCooperativePong)
class LiirPZSimpleSpread(PettingZooSimpleSpread, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PettingZooSimpleSpread)
class LiirPZPursuit(PettingZooPursuit, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PettingZooPursuit)

class LiirPP4P(PressurePlateLinear4P, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PressurePlateLinear4P)
class LiirPP5P(PressurePlateLinear5P, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PressurePlateLinear5P)
class LiirPP6P(PressurePlateLinear6P, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PressurePlateLinear6P)

class LiirRWTiny2Ag(RWarehouseTiny2Ag, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, RWarehouseTiny2Ag)
class LiirRWSmall4Ag(RWarehouseSmall4Ag, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, RWarehouseSmall4Ag)
class LiirRWHard6Ag(RWarehouseHard6Ag, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, RWarehouseHard6Ag)

# StarCraft2: 3m
class LiirSc3M(SC2_3m, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_3m)

# StarCraft2: 27m_vs_30m
class LiirSc27Mvs30M(SC2_27m_vs_30m, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_27m_vs_30m)

# StarCraft2: 3s_vs_5z
class LiirSc3Svs5Z(SC2_3s_vs_5z, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_3s_vs_5z)

# StarCraft2: 3s5z
class LiirSc3S5Z(SC2_3s5z, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_3s5z)

# StarCraft2: 2c_vs_64zg
class LiirSc2Cvs64ZG(SC2_2c_vs_64zg, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_2c_vs_64zg)

# StarCraft2: corridor
class LiirScCorridor(SC2_Corridor, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_Corridor)

# StarCraft2: 5m_vs_6m
class LiirSc5Mvs6M(SC2_5m_vs_6m, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_5m_vs_6m)

# StarCraft2: 6h_vs_8z
class LiirSc6Hvs8Z(SC2_6h_vs_8z, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_6h_vs_8z)

# StarCraft2: MMM2
class LiirScMMM2(SC2_MMM2, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class LiirSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_3s5z_vs_3s6z)

# StarCraft2: 2s_vs_1sc
class LiirSc2Svs1SC(SC2_2s_vs_1sc, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_2s_vs_1sc)

# StarCraft2: 2s3z
class LiirSc2S3Z(SC2_2s3z, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_2s3z)

# StarCraft2: SC2_1c3s5z
class LiirSc1C3S5Z(SC2_1c3s5z, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_1c3s5z)

# StarCraft2: SC2_10m_vs_11m
class LiirSc10Mvs11M(SC2_10m_vs_11m, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_10m_vs_11m)

# StarCraft2: SC2_bane_vs_bane
class LiirScBANEvsBANE(SC2_bane_vs_bane, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, SC2_bane_vs_bane)

class LiirPom4stepSparse(PayoffMatrix4StepSparse, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix4StepSparse)
class LiirPom8stepSparse(PayoffMatrix8StepSparse, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix8StepSparse)
class LiirPom16stepSparse(PayoffMatrix16StepSparse, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix16StepSparse)
class LiirPom32stepSparse(PayoffMatrix32StepSparse, LIIR, General):
    param_overlapped_dict = check_parameters_overlapped(General, LIIR, PayoffMatrix32StepSparse)
######################################################################################################


######################################################################################################
# MATD3

class MATd3MaMujoco2AAnt(MaMujoco2AAnt, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco2AAnt)

class MATd3MaMujoco4AAnt(MaMujoco4AAnt, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco4AAnt)

class MATd3MaMujoco6AHalfcheetah(MaMujoco6AHalfcheetah, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco6AHalfcheetah)

class MATd3MaMujoco3AHopper(MaMujoco3AHopper, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco3AHopper)

class MATd3MaMujoco2AHumanoid(MaMujoco2AHumanoid, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco2AHumanoid)

class MATd3MaMujoco2AHumanoidStandup(MaMujoco2AHumanoidStandup, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujoco2AHumanoidStandup)

class MATd3MaMujocoManyAgentSwimmer(MaMujocoManyAgentSwimmer, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujocoManyAgentSwimmer)

class MATd3MaMujocoCoupledHalfCheetah(MaMujocoCoupledHalfCheetah, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujocoCoupledHalfCheetah)

class MATd3MaMujocoManyAgentAnt(MaMujocoManyAgentAnt, MATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, MATD3, MaMujocoManyAgentAnt)
######################################################################################################


######################################################################################################
# EXPMATD3

class EXPMATD3MaMujoco2AAnt(MaMujoco2AAnt, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco2AAnt)

class EXPMATD3MaMujoco4AAnt(MaMujoco4AAnt, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco4AAnt)

class EXPMATD3MaMujoco6AHalfcheetah(MaMujoco6AHalfcheetah, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco6AHalfcheetah)

class EXPMATD3MaMujoco3AHopper(MaMujoco3AHopper, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco3AHopper)

class EXPMATD3MaMujoco2AHumanoid(MaMujoco2AHumanoid, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco2AHumanoid)

class EXPMATD3MaMujoco2AHumanoidStandup(MaMujoco2AHumanoidStandup, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujoco2AHumanoidStandup)

class EXPMATD3MaMujocoManyAgentSwimmer(MaMujocoManyAgentSwimmer, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujocoManyAgentSwimmer)

class EXPMATD3MaMujocoCoupledHalfCheetah(MaMujocoCoupledHalfCheetah, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujocoCoupledHalfCheetah)

class EXPMATD3MaMujocoManyAgentAnt(MaMujocoManyAgentAnt, EXPMATD3, General):
    param_overlapped_dict = check_parameters_overlapped(General, EXPMATD3, MaMujocoManyAgentAnt)
######################################################################################################

######################################################################################################
# ICES

# StarCraft2: 3m
class ICESSc3M(SC2_3m, ICESQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICESQMIX, SC2_3m)

# StarCraft2: corridor
class ICESScCorridor(SC2_Corridor, ICESQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICESQMIX, SC2_Corridor)

# StarCraft2: 6h_vs_8z
class ICESSc6Hvs8Z(SC2_6h_vs_8z, ICESQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICESQMIX, SC2_6h_vs_8z)

# StarCraft2: MMM2
class ICESScMMM2(SC2_MMM2, ICESQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICESQMIX, SC2_MMM2)

# StarCraft2: 3s5z_vs_3s6z
class ICESSc3S5Zvs3S6Z(SC2_3s5z_vs_3s6z, ICESQMIX, General):
    param_overlapped_dict = check_parameters_overlapped(General, ICESQMIX, SC2_3s5z_vs_3s6z)
######################################################################################################