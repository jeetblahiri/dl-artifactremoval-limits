"""Handcrafted single-channel denoisers.

Each method is imported lazily inside its own factory so that an absent optional
dependency (e.g. ``pywt`` for DWT/SWT, ``PyEMD`` for EMD) does not break import
of the unrelated methods.
"""
from __future__ import annotations

from artifact_limits.methods.handcrafted.bandpass_notch import BandpassNotch
from artifact_limits.methods.handcrafted.identity import Identity
from artifact_limits.methods.handcrafted.savgol import SavGol
from artifact_limits.methods.handcrafted.wiener_filter import WienerFilter
from artifact_limits.methods.handcrafted.wiener_snr_aware import (
    WienerFilterKnownSNR,
    WienerFilterSNRMarginal,
)


def DWTThresholding(*args, **kwargs):
    from artifact_limits.methods.handcrafted.dwt_thresholding import (
        DWTThresholding as _Impl,
    )
    return _Impl(*args, **kwargs)


def SWTThresholding(*args, **kwargs):
    from artifact_limits.methods.handcrafted.swt_thresholding import (
        SWTThresholding as _Impl,
    )
    return _Impl(*args, **kwargs)


def EMDThresholding(*args, **kwargs):
    from artifact_limits.methods.handcrafted.emd_thresholding import (
        EMDThresholding as _Impl,
    )
    return _Impl(*args, **kwargs)


def TotalVariation(*args, **kwargs):
    from artifact_limits.methods.handcrafted.total_variation import (
        TotalVariation as _Impl,
    )
    return _Impl(*args, **kwargs)


__all__ = [
    "Identity", "BandpassNotch", "WienerFilter", "DWTThresholding",
    "SWTThresholding", "EMDThresholding", "TotalVariation", "SavGol",
]
