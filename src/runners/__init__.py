REGISTRY = {}

from .episode_runner import EpisodeRunner
from .episode_exp_runner import ExpEpisodeRunner
REGISTRY["episode"] = EpisodeRunner
REGISTRY["exp_episode"] = ExpEpisodeRunner

from .parallel_runner import ParallelRunner
from .exp_parallel_runner import ExpParallelRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY["exp_parallel"] = ExpParallelRunner

from .noise_parallel_runner import ParallelRunner as NoiseParallelRunner
REGISTRY["noise_parallel"] = NoiseParallelRunner

from .intrinsic_parallel_runner import ParallelRunner as IntrinsicParallelRunner
REGISTRY["intrinsic_parallel"] = IntrinsicParallelRunner

from .util_parallel_runner import ParallelRunner as UtilParallelRunner
REGISTRY["util_parallel"] = UtilParallelRunner

from .surprise_parallel_runner import ParallelRunner as SurpriseParallelRunner
REGISTRY["surprise_parallel"] = SurpriseParallelRunner
