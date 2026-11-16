"""
Audio I/O and preprocessing utilities for SmartSplit.
Handles loading various formats (MP3, WAV, FLAC, M4A) via SoundFile and FFmpeg fallback.
"""

import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


def load_audio(
    audio_path: str | Path,
    target_sr: Optional[int] = None,
    mono: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a NumPy float32 array [-1.0, 1.0].
    Returns (audio_data, sample_rate).
    Shape is (N, 2) for stereo, or (N,) if mono=True.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # First attempt: soundfile (fastest, supports WAV, FLAC, and modern MP3)
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as sf_err:
        # Fallback: FFmpeg decode to raw PCM WAV
        data, sr = _load_via_ffmpeg(path)

    # Ensure shape is (N, 2) for stereo
    if data.ndim == 1:
        if not mono:
            data = np.column_stack([data, data])
    elif data.ndim == 2:
        if mono:
            data = np.mean(data, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2] # take first 2 channels

    # Resample if target_sr is requested and different
    if target_sr is not None and sr != target_sr:
        data, sr = _resample_audio(data, sr, target_sr)

    return data, sr


def _load_via_ffmpeg(path: Path) -> Tuple[np.ndarray, int]:
    """Fallback audio decoder using system FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-ac", "2",
            "-ar", "44100",
            "-f", "wav",
            str(tmp_wav)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to decode audio: {result.stderr.decode(errors='ignore')}")
        data, sr = sf.read(str(tmp_wav), dtype="float32")
        return data, sr
    finally:
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except Exception:
                pass


def _resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Resample audio using scipy.signal.resample_poly."""
    import scipy.signal
    import math

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    if data.ndim == 1:
        resampled = scipy.signal.resample_poly(data, up, down).astype(np.float32)
    else:
        left = scipy.signal.resample_poly(data[:, 0], up, down)
        right = scipy.signal.resample_poly(data[:, 1], up, down)
        min_len = min(len(left), len(right))
        resampled = np.column_stack([left[:min_len], right[:min_len]]).astype(np.float32)

    return resampled, target_sr


def save_audio(
    file_path: str | Path,
    data: np.ndarray,
    sr: int,
    subtype: str = "PCM_16"
) -> Path:
    """Save audio data to disk as WAV."""
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clip to avoid digital clipping distortion
    clipped = np.clip(data, -1.0, 1.0)
    sf.write(str(path), clipped, sr, subtype=subtype)
    return path


def format_duration(seconds: float) -> str:
    """Format duration in seconds to mm:ss."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"
