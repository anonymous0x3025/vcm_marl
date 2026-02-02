REGISTRY = {}

from .rnn_agent import RNNAgent, IntrinsicRNNAgent, IntrinsicBeliefRNNAgent, UtilRNNAgent
from .auto_encoder import AutoEncoder
from .central_rnn_agent import CentralRNNAgent
from .random_network import RandomNetworkModel
from .icm import ICMModule
from .state_decoder import StateDecoder
from .noise_rnn_agent import RNNAgent as NoiseRNNAgent
from .vae_rnn_agent import VAERNNAgent
from .vae_intrinsic_agent import VAEIntrinsicAgent
from .rnn_fast_agent import RNNFastAgent
from .surprise_agent import SurpriseAgent
from .intrinsic_value_critic import IntrinsicValueCritic
from .central_rnn_agent import IntrinsicBeliefCentralRNNAgent
from .n_rnn_agent import NRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_fast"] = RNNFastAgent
REGISTRY["util_rnn"] = UtilRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["auto_encoder"] = AutoEncoder
REGISTRY["random_network"] = RandomNetworkModel
REGISTRY["icm_module"] = ICMModule
REGISTRY["state_decoder"] = StateDecoder
REGISTRY["noise_rnn"] = NoiseRNNAgent
REGISTRY["vae_rnn"] = VAERNNAgent
REGISTRY["vae_intrinsic_rnn"] = VAEIntrinsicAgent
REGISTRY["intrinsic_rnn"] = IntrinsicRNNAgent
REGISTRY["intrinsic_rnn_belief"] = IntrinsicBeliefRNNAgent
REGISTRY["surprise_rnn"] = SurpriseAgent
REGISTRY["intrinsic_critic_rnn"] = IntrinsicValueCritic
REGISTRY["intrinsic_central_rnn_belief"] = IntrinsicBeliefCentralRNNAgent
REGISTRY["n_rnn"] = NRNNAgent
