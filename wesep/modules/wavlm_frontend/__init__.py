"""WavLM frontend stack for wesep.

This package hosts a self-contained copy of the WavLM / wav2vec2 building
blocks originally used in `wespeaker_hubert`, stripped of any public pruning
APIs. It is used to provide SSL features and layer-wise representations for
speaker-conditioned separation models.
"""

from .model import Wav2Vec2Model, wavlm_base  # noqa: F401
from .frontend import HuggingfaceFrontendWavLM, WavLMFrontendConfig  # noqa: F401

