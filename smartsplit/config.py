"""
Configuration settings for SmartSplit MCP server.
"""

import os
from pathlib import Path

# Base Paths
DEFAULT_OUTPUT_DIR = Path(os.environ.get("SMARTSPLIT_OUTPUT_DIR", Path.home() / ".smartsplit" / "output"))
DEFAULT_CACHE_DIR = Path(os.environ.get("SMARTSPLIT_CACHE_DIR", Path.home() / ".smartsplit" / "cache"))

# Ensure base directories exist
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Audio Processing Settings
TARGET_SAMPLE_RATE = 44100
STFT_NPERSEG = 2048
STFT_NOVERLAP = 1536
STFT_WINDOW = "hann"

# Frequency Band Definitions (Hz)
FREQ_SUB_BASS_MAX = 60
FREQ_BASS_MIN = 60
FREQ_BASS_MAX = 250
FREQ_LOW_MID_MIN = 250
FREQ_LOW_MID_MAX = 1000
FREQ_HIGH_MID_MIN = 1000
FREQ_HIGH_MID_MAX = 4000
FREQ_TREBLE_MIN = 4000
FREQ_TREBLE_MAX = 8000
FREQ_AIR_MIN = 8000

# Vocal Band definition for center cancellation
VOCAL_BAND_LOW = 110
VOCAL_BAND_HIGH = 8500

# Demucs Settings
DEFAULT_DEMUCS_MODEL = os.environ.get("SMARTSPLIT_DEMUCS_MODEL", "htdemucs")
DEMUCS_DEVICE = os.environ.get("SMARTSPLIT_DEVICE", "cpu")
