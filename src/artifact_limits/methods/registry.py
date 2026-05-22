"""Method registry: a single dict-of-factories used by experiments and CLIs.

Add a new method by registering its class under a stable string ID. Callers
construct an instance via ``build_method("dwt", level=4)`` which is more
ergonomic than importing each class directly.
"""
from __future__ import annotations

from typing import Callable

from artifact_limits.methods.base import DenoiserBase
from artifact_limits.methods.handcrafted import (
    BandpassNotch,
    DWTThresholding,
    EMDThresholding,
    Identity,
    SavGol,
    SWTThresholding,
    TotalVariation,
    WienerFilter,
    WienerFilterKnownSNR,
    WienerFilterSNRMarginal,
)


def _torch_factory(name: str) -> Callable[..., DenoiserBase]:
    """Defer torch imports until a deep model is actually constructed."""
    def _build(**kwargs) -> DenoiserBase:
        from artifact_limits.methods.deep import (
            DeepSeparator, EEGDenoiseFormer, ICUNet, NovelCNN, RNNLSTM, SimpleCNN,
        )
        from artifact_limits.methods.deep.ddpm_eeg import DDPMEEG
        mapping = {
            "simple_cnn": SimpleCNN,
            "novel_cnn": NovelCNN,
            "rnn_lstm": RNNLSTM,
            "deep_separator": DeepSeparator,
            "eeg_denoise_former": EEGDenoiseFormer,
            "ic_unet": ICUNet,
            "ddpm_eeg": DDPMEEG,
        }
        return mapping[name](**kwargs)
    return _build


_REGISTRY: dict[str, Callable[..., DenoiserBase]] = {
    # Handcrafted
    "identity": Identity,
    "bandpass_notch": BandpassNotch,
    "wiener_filter": WienerFilter,
    "wiener_known_snr": WienerFilterKnownSNR,
    "wiener_snr_marginal": WienerFilterSNRMarginal,
    "dwt": DWTThresholding,
    "swt": SWTThresholding,
    "emd": EMDThresholding,
    "total_variation": TotalVariation,
    "savgol": SavGol,
    # Deep (lazy)
    "simple_cnn": _torch_factory("simple_cnn"),
    "novel_cnn": _torch_factory("novel_cnn"),
    "rnn_lstm": _torch_factory("rnn_lstm"),
    "deep_separator": _torch_factory("deep_separator"),
    "eeg_denoise_former": _torch_factory("eeg_denoise_former"),
    "ic_unet": _torch_factory("ic_unet"),
    "ddpm_eeg": _torch_factory("ddpm_eeg"),
}

HANDCRAFTED_IDS = ("identity", "bandpass_notch", "wiener_filter",
                   "wiener_known_snr", "wiener_snr_marginal",
                   "dwt", "swt", "emd", "total_variation", "savgol")
DEEP_IDS = ("simple_cnn", "novel_cnn", "rnn_lstm", "deep_separator",
            "eeg_denoise_former", "ic_unet")


def list_methods() -> list[str]:
    return list(_REGISTRY.keys())


def build_method(method_id: str, **kwargs) -> DenoiserBase:
    if method_id not in _REGISTRY:
        raise KeyError(f"Unknown method id {method_id!r}. Available: {list_methods()}")
    return _REGISTRY[method_id](**kwargs)
