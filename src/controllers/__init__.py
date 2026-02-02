REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_intrinsic_controller import BasicIntrinsicMAC
from .exp_controller import ExpMAC
from .central_basic_controller import CentralBasicMAC
from .rnd_controller import RndMAC
from .icm_controller import IcmMAC
from .noise_controller import NoiseMAC
from .vi_util_exp_controller import VIUtilExpMAC
from .vi_intrinsic_controller import VIIntrinsicMAC
from .fast_controller import FastMAC
from .surprise_controller import SurpriseMAC
from .intrinisc_critic_controller import IntrinsicCriticMAC
from .ices_n_controller import ICESNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_intrinsic_mac"] = BasicIntrinsicMAC
REGISTRY["exp_mac"] = ExpMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["rnd_mac"] = RndMAC
REGISTRY["icm_mac"] = IcmMAC
REGISTRY["noise_mac"] = NoiseMAC
REGISTRY["vi_util_exp_mac"] = VIUtilExpMAC
REGISTRY["vi_intrinsic_mac"] = VIIntrinsicMAC
REGISTRY["fast_mac"] = FastMAC
REGISTRY["surprise_mac"] = SurpriseMAC
REGISTRY["intrinsic_critic_mac"] = IntrinsicCriticMAC
REGISTRY["ices_n_mac"] = ICESNMAC
