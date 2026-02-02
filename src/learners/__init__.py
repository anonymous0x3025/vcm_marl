from .q_learner import QLearner
from .exp_q_learner import ExpQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .exp_qtran_learner import ExpQLearner as ExpQTranLearner
from .exp_matd3_learner import ExpMATd3Learner
from .matd3_learner import MATd3Learner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .exp_dmaq_qatten_learner import ExpDMAQ_qattenLearner
from .max_q_learner import MAXQLearner
from .exp_max_q_learner import ExpMAXQLearner
from .semi_emc_q_learner import SemiEMCQLearner
from .rnd_q_learner import RNDQLearner
from .icm_q_learner import IcmQLearner
from .noise_q_learner import QLearner as NoiseQLearner
from .vi_exp_q_learner import VIExpQLearner
from .vi_util_exp_q_learner import VIUtilExpQLearner
from .vi_intrinsic_q_learner import VIIntrinsicQLearner
from .q_learner_for_intrinsic import QLearnerForIntrinsic
from .vi_intrinsic_qplex_learner import VIIntrinsicQplexLearner
from .qplex_curiosity_vdn_learner import QPLEX_curiosity_vdn_Learner
from .surprise_q_learner import SurpriseQLearner
from .liir_learner import LIIRLearner
from .vi_intrinsic_max_learner import VIIntrinsicMAXQLearner
from .vi_intrinsic_qtran_learner import VIIntrinsicQtranLearner
from .ices_nq_learner import ICESNQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["exp_q_learner"] = ExpQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["exp_qtran_learner"] = ExpQTranLearner
REGISTRY["matd3_learner"] = MATd3Learner
REGISTRY["exp_matd3_learner"] = ExpMATd3Learner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["exp_dmaq_qatten_learner"] = ExpDMAQ_qattenLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["exp_max_q_learner"] = ExpMAXQLearner
REGISTRY["semi_emc_q_learner"] = SemiEMCQLearner
REGISTRY["rnd_q_learner"] = RNDQLearner
REGISTRY["icm_q_learner"] = IcmQLearner
REGISTRY["noise_q_learner"] = NoiseQLearner
REGISTRY["vi_exp_q_learner"] = VIExpQLearner
REGISTRY["vi_util_exp_q_learner"] = VIUtilExpQLearner
REGISTRY["vi_intrinsic_q_learner"] = VIIntrinsicQLearner
REGISTRY["vi_intrinsic_qplex_learner"] = VIIntrinsicQplexLearner
REGISTRY["qplex_curiosity_vdn_learner"] = QPLEX_curiosity_vdn_Learner
REGISTRY["surprise_q_learner"] = SurpriseQLearner
REGISTRY["liir_learner"] = LIIRLearner
REGISTRY["vi_intrinsic_max_learner"] = VIIntrinsicMAXQLearner
REGISTRY["vi_intrinsic_qtran_learner"] = VIIntrinsicQtranLearner
REGISTRY["ices_nq_learner"] = ICESNQLearner
