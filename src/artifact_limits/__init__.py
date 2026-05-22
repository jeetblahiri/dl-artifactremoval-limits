"""artifact_limits — empirical and theoretical limits of EEG artifact removal on EEGdenoiseNet."""

__version__ = "0.1.0"

SEEDS = (42, 123, 2024, 7, 31337)
SAMPLING_FREQ = 256
SEGMENT_LEN = 512
SNR_DB_LEVELS = (-7, -4, -1, 2)
ARTIFACT_TYPES = ("EOG", "EMG")
