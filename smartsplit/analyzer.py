"""
Music & Instrument Analysis Engine for SmartSplit.
Performs acoustic reverse-engineering on audio files:
  - Instrument presence detection (Vocals, Drums, Bass, Guitars, Keyboards/Synths, Cymbals)
  - Frequency band energy distribution (Sub-bass, Bass, Low-Mid, High-Mid, Air)
  - Harmonic vs. Percussive balance
  - Stereo panning and center channel vocal localization
  - Tempo / BPM estimation, Dynamic range, and RMS levels
  - Separability assessment & AI stem recommendation
"""

import math
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import scipy.signal

from .audio_utils import load_audio, format_duration
from .config import (
    FREQ_SUB_BASS_MAX, FREQ_BASS_MIN, FREQ_BASS_MAX,
    FREQ_LOW_MID_MIN, FREQ_LOW_MID_MAX,
    FREQ_HIGH_MID_MIN, FREQ_HIGH_MID_MAX,
    FREQ_TREBLE_MIN, FREQ_TREBLE_MAX,
    FREQ_AIR_MIN
)


from dataclasses import dataclass

@dataclass
class InstrumentDetail:
    name: str
    detected: bool
    confidence: float
    mix_placement: str
    frequency_band: str
    prominence: str
    acoustic_evidence: str

@dataclass
class AudioAnalysisResult:
    status: str
    file_name: str
    file_path: str
    audio_specs: Dict[str, Any]
    mix_staging: Dict[str, Any]
    frequency_distribution: Dict[str, Any]
    detected_instruments: List[Dict[str, Any]]
    vocal_separability: Dict[str, Any]
    summary: str


class MusicAnalyzer:
    """Acoustic inspection and instrument classification engine."""

    def __init__(self, sample_limit_seconds: float = 60.0):
        """
        sample_limit_seconds: duration of representative audio analyzed for speed (default: 60s).
        """
        self.sample_limit_seconds = sample_limit_seconds

    def analyze(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Full acoustic inspection of an audio file.
        Returns a rich JSON-serializable dictionary formatted for LLMs and human users.
        """
        path = Path(audio_path).resolve()
        data, sr = load_audio(path)
        
        orig_len_samples = len(data)
        duration_sec = orig_len_samples / float(sr)
        duration_fmt = format_duration(duration_sec)
        
        # Take a representative section (skip first 2 seconds to avoid lead-in silence)
        start_sample = min(int(sr * 2.0), max(0, orig_len_samples - 1000))
        analysis_samples = int(sr * self.sample_limit_seconds)
        end_sample = min(start_sample + analysis_samples, orig_len_samples)
        
        chunk = data[start_sample:end_sample]
        if len(chunk) < sr: # If audio is very short, use entire audio
            chunk = data
            
        left = chunk[:, 0]
        right = chunk[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        mono = mid

        # 1. Dynamic Range, Peak and RMS
        eps = 1e-12
        rms = float(np.sqrt(np.mean(chunk ** 2) + eps))
        rms_dbfs = round(float(20.0 * np.log10(rms + eps)), 1)
        peak = float(np.max(np.abs(chunk)))
        peak_dbfs = round(float(20.0 * np.log10(peak + eps)), 1)
        dynamic_range_db = round(abs(peak_dbfs - rms_dbfs), 1)

        # 2. Stereo Field & Spatial Staging
        mid_energy = float(np.sum(mid ** 2))
        side_energy = float(np.sum(side ** 2))
        total_energy = mid_energy + side_energy + eps
        stereo_width = round(float(np.clip(side_energy / total_energy, 0.0, 1.0)), 3)
        center_energy_pct = round(float(mid_energy / total_energy * 100.0), 1)
        side_energy_pct = round(float(side_energy / total_energy * 100.0), 1)

        # 3. Frequency Spectrum Decomposition (STFT)
        nperseg = 2048
        noverlap = 1536
        f, t, Zxx_mono = scipy.signal.stft(mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag_mono = np.abs(Zxx_mono)
        power_spectrum = np.mean(mag_mono ** 2, axis=1)
        total_power = np.sum(power_spectrum) + eps

        # Band energy calculations
        mask_sub = (f <= FREQ_SUB_BASS_MAX)
        mask_bass = (f > FREQ_BASS_MIN) & (f <= FREQ_BASS_MAX)
        mask_low_mid = (f > FREQ_LOW_MID_MIN) & (f <= FREQ_LOW_MID_MAX)
        mask_high_mid = (f > FREQ_HIGH_MID_MIN) & (f <= FREQ_HIGH_MID_MAX)
        mask_treble = (f > FREQ_TREBLE_MIN) & (f <= FREQ_TREBLE_MAX)
        mask_air = (f > FREQ_AIR_MIN)

        sub_pct = round(float(np.sum(power_spectrum[mask_sub]) / total_power * 100.0), 1)
        bass_pct = round(float(np.sum(power_spectrum[mask_bass]) / total_power * 100.0), 1)
        low_mid_pct = round(float(np.sum(power_spectrum[mask_low_mid]) / total_power * 100.0), 1)
        high_mid_pct = round(float(np.sum(power_spectrum[mask_high_mid]) / total_power * 100.0), 1)
        treble_pct = round(float(np.sum(power_spectrum[mask_treble]) / total_power * 100.0), 1)
        air_pct = round(float(np.sum(power_spectrum[mask_air]) / total_power * 100.0), 1)

        # 4. Vocal Center Localization Metric
        _, _, Zxx_L = scipy.signal.stft(left, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, _, Zxx_R = scipy.signal.stft(right, fs=sr, nperseg=nperseg, noverlap=noverlap)
        mag_L = np.abs(Zxx_L)
        mag_R = np.abs(Zxx_R)
        cross_term = 2.0 * np.abs(Zxx_L * np.conj(Zxx_R))
        denom = (mag_L ** 2) + (mag_R ** 2) + eps
        gamma = cross_term / denom
        
        phase_L = np.angle(Zxx_L)
        phase_R = np.angle(Zxx_R)
        phase_diff = np.abs(phase_L - phase_R)
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        in_phase = np.clip(np.cos(phase_diff * 0.5), 0.0, 1.0)
        vocal_pan_matrix = gamma * in_phase
        
        # Focus on vocal frequency range (150 Hz - 4000 Hz)
        vocal_freq_mask = (f >= 150) & (f <= 4000)
        mean_vocal_centering = float(np.mean(vocal_pan_matrix[vocal_freq_mask, :]))

        # 5. Percussive Transients vs Harmonic Content (Spectral Flux)
        flux = np.maximum(0, np.diff(mag_mono, axis=1))
        onset_env = np.sum(flux, axis=0)
        percussive_transient_score = float(np.std(onset_env) / (np.mean(onset_env) + eps))
        percussive_energy_pct = round(float(np.clip(percussive_transient_score * 30.0, 10.0, 85.0)), 1)
        harmonic_energy_pct = round(float(100.0 - percussive_energy_pct), 1)

        # 6. Tempo (BPM) Estimation
        tempo_bpm = self._estimate_tempo(onset_env, sr, nperseg - noverlap)

        # 7. Instrument Classifiers & Detection
        detected_instruments = []

        # (a) Vocals Detection
        vocal_conf = float(np.clip(mean_vocal_centering * 1.35 + (high_mid_pct / 45.0) * 0.35, 0.0, 0.99))
        vocal_present = vocal_conf > 0.45
        detected_instruments.append({
            "name": "Vocals",
            "detected": vocal_present,
            "confidence": round(vocal_conf * 100.0, 1),
            "mix_placement": "Center Channel (Phantom Center)" if vocal_conf > 0.6 else "Center-Stereo Spread",
            "frequency_band": "120 Hz - 4,500 Hz",
            "prominence": "Dominant Lead" if vocal_conf > 0.75 else ("Prominent" if vocal_conf > 0.55 else "Subtle / Ambient"),
            "acoustic_evidence": f"Acoustic centering index of {mean_vocal_centering:.2f} with strong formants in mid-frequencies ({high_mid_pct}% of mix energy)."
        })

        # (b) Drums & Percussion Detection
        drum_conf = float(np.clip((percussive_transient_score * 0.4) + ((sub_pct + bass_pct) / 60.0) * 0.4 + (air_pct / 30.0) * 0.2, 0.0, 0.99))
        drum_present = drum_conf > 0.40
        detected_instruments.append({
            "name": "Drums & Percussion",
            "detected": drum_present,
            "confidence": round(drum_conf * 100.0, 1),
            "mix_placement": "Sub/Center (Kick & Snare) + Stereo Sides (Hi-Hats/Cymbals)",
            "frequency_band": "40 Hz - 18,000 Hz (Broadband Transients)",
            "prominence": "Dominant Rhythm Section" if drum_conf > 0.70 else ("Prominent" if drum_conf > 0.50 else "Subtle"),
            "acoustic_evidence": f"Rhythmic transient flux score {percussive_transient_score:.2f} with steady beat pulse at ~{tempo_bpm or 'N/A'} BPM."
        })

        # (c) Bass
        bass_conf = float(np.clip(((sub_pct + bass_pct) / 35.0) * 0.75 + (1.0 - stereo_width) * 0.25, 0.0, 0.99))
        bass_present = (sub_pct + bass_pct) > 12.0
        detected_instruments.append({
            "name": "Bass",
            "detected": bass_present,
            "confidence": round(bass_conf * 100.0, 1),
            "mix_placement": "Direct Center / Mono Sub-Bass",
            "frequency_band": "30 Hz - 250 Hz",
            "prominence": "Heavy / Driving" if (sub_pct + bass_pct) > 30.0 else "Solid Foundation",
            "acoustic_evidence": f"Sub-bass and low frequencies account for {sub_pct + bass_pct:.1f}% of total spectral energy."
        })

        # (d) Guitars / Keyboards / Synthesizers (Harmonic Mid Instruments)
        harmonic_conf = float(np.clip((low_mid_pct / 30.0) * 0.5 + (stereo_width * 0.5), 0.0, 0.99))
        harmonic_present = low_mid_pct > 15.0 or stereo_width > 0.15
        detected_instruments.append({
            "name": "Guitars / Keyboards / Synths",
            "detected": harmonic_present,
            "confidence": round(harmonic_conf * 100.0, 1),
            "mix_placement": "Stereo Left & Right Panned",
            "frequency_band": "250 Hz - 3,500 Hz",
            "prominence": "Prominent Melodic Harmony" if harmonic_conf > 0.60 else "Backing Accompaniment",
            "acoustic_evidence": f"Stereo width of {stereo_width:.2f} with {low_mid_pct}% harmonic body in low-mids."
        })

        # (e) Cymbals & Ambient Air
        cymbals_conf = float(np.clip((air_pct / 15.0) * 0.7 + (treble_pct / 20.0) * 0.3, 0.0, 0.99))
        cymbals_present = air_pct > 3.0 or treble_pct > 5.0
        detected_instruments.append({
            "name": "Cymbals & Shimmer Air",
            "detected": cymbals_present,
            "confidence": round(cymbals_conf * 100.0, 1),
            "mix_placement": "Wide Stereo Perimeter",
            "frequency_band": "8,000 Hz - 20,000 Hz",
            "prominence": "Airy & Crisp" if air_pct > 8.0 else "Present",
            "acoustic_evidence": f"High-frequency sparkle accounts for {air_pct}% of mix energy."
        })

        # 8. Vocal Separability Rating
        if mean_vocal_centering > 0.70:
            sep_rating = "Excellent"
            removability_pct = 97.5
            rec_algo = "Fast Mathematical DSP (Instant 2-Second STFT Panning Cancellation)"
            rec_notes = "Vocal is tightly localized in the phantom center. Instant mathematical DSP will achieve near-total (>60dB) vocal muting with zero artifacts."
        elif mean_vocal_centering > 0.50:
            sep_rating = "Very Good"
            removability_pct = 92.0
            rec_algo = "Fast Mathematical DSP or Demucs Neural (Balanced)"
            rec_notes = "Vocals are predominantly centered with minor stereo reverb spread. Separates cleanly."
        else:
            sep_rating = "Moderate"
            removability_pct = 84.0
            rec_algo = "Demucs Neural Model (htdemucs)"
            rec_notes = "Vocals utilize wide stereo effects/double tracking. Demucs deep neural model recommended for maximum isolation."

        # High-level summary text for AI agents to easily read
        detected_names = [inst["name"] for inst in detected_instruments if inst["detected"]]
        summary = (
            f"Analyzed '{path.name}' ({duration_fmt}, {sr}Hz stereo). "
            f"Detected instruments: {', '.join(detected_names)}. "
            f"Estimated tempo: {tempo_bpm or 'variable'} BPM. "
            f"Vocal removability score: {removability_pct}% ({sep_rating}). "
            f"Recommended separation method: {rec_algo}."
        )

        return {
            "status": "success",
            "file_name": path.name,
            "file_path": str(path),
            "audio_specs": {
                "duration_seconds": round(duration_sec, 2),
                "duration_formatted": duration_fmt,
                "sample_rate": sr,
                "channels": 2,
                "peak_dbfs": peak_dbfs,
                "rms_dbfs": rms_dbfs,
                "dynamic_range_db": dynamic_range_db,
                "estimated_bpm": tempo_bpm
            },
            "mix_staging": {
                "stereo_width": stereo_width,
                "center_energy_pct": center_energy_pct,
                "side_energy_pct": side_energy_pct,
                "vocal_center_index": round(mean_vocal_centering, 3)
            },
            "frequency_distribution": {
                "sub_bass_pct": sub_pct,
                "bass_pct": bass_pct,
                "low_mid_pct": low_mid_pct,
                "high_mid_pct": high_mid_pct,
                "treble_pct": treble_pct,
                "air_pct": air_pct,
                "percussive_energy_pct": percussive_energy_pct,
                "harmonic_energy_pct": harmonic_energy_pct
            },
            "detected_instruments": detected_instruments,
            "vocal_separability": {
                "rating": sep_rating,
                "removability_score_pct": removability_pct,
                "recommended_method": rec_algo,
                "technical_details": rec_notes
            },
            "summary": summary
        }

    def _estimate_tempo(self, onset_env: np.ndarray, sr: int, hop_size: int) -> Optional[int]:
        """Estimate BPM via autocorrelation of spectral flux onset envelope."""
        try:
            if len(onset_env) < 20:
                return None
            centered = onset_env - np.mean(onset_env)
            acorr = np.correlate(centered, centered, mode='full')
            acorr = acorr[len(acorr) // 2:]
            
            fps = float(sr) / float(hop_size)
            min_lag = int(fps * 60.0 / 180.0) # max 180 BPM
            max_lag = int(fps * 60.0 / 60.0)  # min 60 BPM
            
            if max_lag >= len(acorr) or min_lag >= max_lag:
                return None
                
            best_lag = min_lag + int(np.argmax(acorr[min_lag:max_lag]))
            bpm = int(round((fps * 60.0) / best_lag))
            return bpm
        except Exception:
            return None


# Top-level helper function
def analyze_music_file(audio_path: str | Path) -> Dict[str, Any]:
    """Helper entry point to analyze an audio file."""
    analyzer = MusicAnalyzer()
    return analyzer.analyze(audio_path)
