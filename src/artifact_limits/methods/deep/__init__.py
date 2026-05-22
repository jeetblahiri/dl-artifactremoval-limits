"""Deep-learning denoisers (PyTorch).

All architectures share a thin sklearn-style interface via `TorchDenoiserBase`,
so that they slot into the registry alongside the handcrafted methods.
"""
from artifact_limits.methods.deep.base_denoiser import TorchDenoiserBase  # noqa: F401
from artifact_limits.methods.deep.deep_separator import DeepSeparator  # noqa: F401
from artifact_limits.methods.deep.eeg_denoise_former import EEGDenoiseFormer  # noqa: F401
from artifact_limits.methods.deep.ic_unet import ICUNet  # noqa: F401
from artifact_limits.methods.deep.novel_cnn import NovelCNN  # noqa: F401
from artifact_limits.methods.deep.rnn_lstm import RNNLSTM  # noqa: F401
from artifact_limits.methods.deep.simple_cnn import SimpleCNN  # noqa: F401
