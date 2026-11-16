"""
Stem Separation Engine for SmartSplit MCP.
Combines ultra-fast Mathematical Center-Channel DSP Panning Cancellation (<2s)
and Deep Neural Demucs (htdemucs) for studio-grade isolation.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_DEMUCS_MODEL, DEMUCS_DEVICE
from .dsp_math import fast_spectral_separation, isolate_bass_dsp, separate_harmonic_percussive_dsp
from .audio_utils import load_audio, save_audio, format_duration


@dataclass
class SeparationResult:
    status: str
    source_file: str
    source_path: str
    stems: Dict[str, str]
    algorithm_used: str
    quality_profile: str
    processing_time_seconds: float
    output_directory: str
    summary: str


class StemSeparator:
    """Multi-tiered separation engine supporting fast DSP and Demucs neural separation."""

    def __init__(self, default_output_dir: Optional[Path] = None):
        self.output_dir = Path(default_output_dir) if default_output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def separate_vocals(
        self,
        audio_path: str | Path,
        quality: str = "fast",
        output_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Extracts lead vocal stem and instrumental backing track.
        quality: 'fast' (Instant 2-sec DSP) or 'balanced' / 'high' (Demucs Neural).
        """
        start_time = time.time()
        source = Path(audio_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        track_slug = source.stem.replace(" ", "_")
        target_dir = Path(output_dir).resolve() if output_dir else (self.output_dir / track_slug)
        target_dir.mkdir(parents=True, exist_ok=True)

        vocal_target = target_dir / f"{track_slug}_vocals.wav"
        inst_target = target_dir / f"{track_slug}_instrumental.wav"

        algo_used = "Mathematical Center-Channel Panning Cancellation + STFT Wiener Masking"

        if quality.lower() == "fast":
            success = fast_spectral_separation(source, vocal_target, inst_target)
            if not success:
                raise RuntimeError("Mathematical spectral separation failed.")
        else:
            # Attempt Demucs separation
            try:
                self._run_demucs(source, target_dir, two_stems=True)
                algo_used = f"Facebook Demucs Neural Network ({DEFAULT_DEMUCS_MODEL})"
            except Exception as demucs_err:
                print(f"Demucs unavailable or failed ({demucs_err}), falling back to Mathematical DSP...")
                success = fast_spectral_separation(source, vocal_target, inst_target)
                if not success:
                    raise RuntimeError(f"Both Demucs and DSP fallback failed: {demucs_err}")
                algo_used = "Mathematical DSP (Fallback after Demucs error)"

        elapsed = round(time.time() - start_time, 2)
        summary = (
            f"Vocal separation completed in {elapsed}s using {algo_used}. "
            f"Clean instrumental and vocal stems saved to {target_dir}."
        )

        return {
            "status": "success",
            "source_file": source.name,
            "source_path": str(source),
            "algorithm_used": algo_used,
            "quality_profile": quality,
            "processing_time_seconds": elapsed,
            "output_directory": str(target_dir),
            "stems": {
                "instrumental": str(inst_target),
                "vocals": str(vocal_target)
            },
            "summary": summary
        }

    def separate_stems(
        self,
        audio_path: str | Path,
        stems: int = 2,
        quality: str = "fast",
        output_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Separates into 2 stems (vocals/instrumental) or 4 stems (vocals/drums/bass/other).
        """
        if stems == 2:
            return self.separate_vocals(audio_path, quality=quality, output_dir=output_dir)

        start_time = time.time()
        source = Path(audio_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        track_slug = source.stem.replace(" ", "_")
        target_dir = Path(output_dir).resolve() if output_dir else (self.output_dir / f"{track_slug}_4stems")
        target_dir.mkdir(parents=True, exist_ok=True)

        vocal_target = target_dir / f"{track_slug}_vocals.wav"
        drums_target = target_dir / f"{track_slug}_drums.wav"
        bass_target = target_dir / f"{track_slug}_bass.wav"
        other_target = target_dir / f"{track_slug}_other.wav"

        algo_used = ""
        demucs_success = False
        
        # Only invoke Demucs when quality is 'balanced', 'high', or 'deep'
        if quality.lower() not in ["fast", "instant"]:
            try:
                self._run_demucs(source, target_dir, two_stems=False)
                algo_used = f"Demucs Neural Network 4-Stem Model ({DEFAULT_DEMUCS_MODEL})"
                demucs_success = True
            except Exception as demucs_err:
                print(f"Demucs 4-stem failed ({demucs_err}), falling back to DSP filtering...")
                demucs_success = False

        if not demucs_success:
            # DSP 4-stem reconstruction:
            # 1. Vocal vs Instrumental
            fast_spectral_separation(source, vocal_target, target_dir / "temp_inst.wav")
            data, sr = load_audio(source)
            
            # 2. Bass: lowpass filter < 220Hz
            bass_data = isolate_bass_dsp(data, sr, cutoff_hz=220.0)
            save_audio(bass_target, bass_data, sr)

            # 3. Drums: Transient percussive extraction via HPSS
            _, drums_data = separate_harmonic_percussive_dsp(data, sr)
            save_audio(drums_target, drums_data, sr)

            # 4. Other (Melodic mid-range harmonic accompaniment)
            temp_inst_path = target_dir / "temp_inst.wav"
            if temp_inst_path.exists():
                inst_data, _ = load_audio(temp_inst_path)
                other_data = inst_data - (bass_data[:len(inst_data)] * 0.7)
                save_audio(other_target, other_data, sr)
                try:
                    temp_inst_path.unlink()
                except Exception:
                    pass
            else:
                save_audio(other_target, data - bass_data, sr)

            algo_used = "Mathematical DSP 4-Way Spectral Decomposition (Zero-Latency DSP)"

        elapsed = round(time.time() - start_time, 2)
        summary = (
            f"4-stem separation completed in {elapsed}s using {algo_used}. "
            f"Stems: Vocals, Drums, Bass, and Other/Harmony."
        )

        return {
            "status": "success",
            "source_file": source.name,
            "source_path": str(source),
            "algorithm_used": algo_used,
            "stems_count": 4,
            "processing_time_seconds": elapsed,
            "output_directory": str(target_dir),
            "stems": {
                "vocals": str(vocal_target),
                "drums": str(drums_target),
                "bass": str(bass_target),
                "other": str(other_target)
            },
            "summary": summary
        }

    def isolate_instrument(
        self,
        audio_path: str | Path,
        target_instrument: str = "vocals",
        quality: str = "fast",
        output_dir: Optional[str | Path] = None
    ) -> Dict[str, Any]:
        """
        Isolate a single specific instrument ('vocals', 'instrumental', 'drums', 'bass', or 'other').
        Runs with maximum efficiency by isolating only the requested target.
        """
        start_time = time.time()
        source = Path(audio_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        target = target_instrument.lower().strip()
        track_slug = source.stem.replace(" ", "_")
        target_dir = Path(output_dir).resolve() if output_dir else (self.output_dir / track_slug)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Vocals or Instrumental
        if target in ["vocal", "vocals", "instrumental", "backing", "music", "karaoke"]:
            res = self.separate_vocals(audio_path, quality=quality, output_dir=target_dir)
            stem_key = "vocals" if "vocal" in target else "instrumental"
            return {
                "status": "success",
                "isolated_instrument": stem_key,
                "file_path": res["stems"][stem_key],
                "all_stems": res["stems"],
                "algorithm_used": res["algorithm_used"],
                "processing_time_seconds": res["processing_time_seconds"],
                "summary": f"Successfully isolated '{stem_key}' stem from {source.name}."
            }

        # 2. Bass Fundamental Isolation (Direct Butterworth lowpass < 220Hz)
        if target in ["bass", "sub", "bassline"]:
            data, sr = load_audio(source)
            bass_data = isolate_bass_dsp(data, sr, cutoff_hz=220.0)
            bass_path = target_dir / f"{track_slug}_bass.wav"
            save_audio(bass_path, bass_data, sr)
            elapsed = round(time.time() - start_time, 2)
            return {
                "status": "success",
                "isolated_instrument": "bass",
                "file_path": str(bass_path),
                "algorithm_used": "Mathematical Zero-Phase 4th-Order Butterworth Low-Pass Filter (<220Hz)",
                "processing_time_seconds": elapsed,
                "summary": f"Successfully extracted isolated bass track in {elapsed}s to: {bass_path}"
            }

        # 3. Drums / Percussion Isolation (Fitzgerald HPSS Transient Matrix)
        if target in ["drums", "drum", "percussion", "beat"]:
            data, sr = load_audio(source)
            _, drums_data = separate_harmonic_percussive_dsp(data, sr)
            drums_path = target_dir / f"{track_slug}_drums.wav"
            save_audio(drums_path, drums_data, sr)
            elapsed = round(time.time() - start_time, 2)
            return {
                "status": "success",
                "isolated_instrument": "drums",
                "file_path": str(drums_path),
                "algorithm_used": "Fitzgerald HPSS Percussive Transient Extraction",
                "processing_time_seconds": elapsed,
                "summary": f"Successfully extracted isolated drums & percussion in {elapsed}s to: {drums_path}"
            }

        # 4. Other / Harmony Accompaniment
        if target in ["other", "harmony", "guitar", "keys", "piano", "synth"]:
            res_4 = self.separate_stems(audio_path, stems=4, quality=quality, output_dir=target_dir)
            return {
                "status": "success",
                "isolated_instrument": "other",
                "file_path": res_4["stems"]["other"],
                "all_stems": res_4["stems"],
                "algorithm_used": res_4["algorithm_used"],
                "processing_time_seconds": res_4["processing_time_seconds"],
                "summary": f"Successfully extracted harmonic accompaniment to: {res_4['stems']['other']}"
            }

        raise ValueError(f"Unknown target instrument '{target_instrument}'. Supported: vocals, instrumental, drums, bass, other.")

    def _run_demucs(self, source_path: Path, output_dir: Path, two_stems: bool = True):
        """Invoke Demucs CLI sub-process."""
        demucs_temp = output_dir / "demucs_temp"
        demucs_temp.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m", "demucs.separate",
            "-d", DEMUCS_DEVICE,
            "-n", DEFAULT_DEMUCS_MODEL,
            "-o", str(demucs_temp)
        ]
        if two_stems:
            cmd.extend(["--two-stems", "vocals"])
        cmd.append(str(source_path))

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if proc.returncode != 0:
            shutil.rmtree(demucs_temp, ignore_errors=True)
            raise RuntimeError(f"Demucs returned exit code {proc.returncode}: {proc.stderr.decode(errors='ignore')}")

        track_slug = source_path.stem
        model_out = demucs_temp / DEFAULT_DEMUCS_MODEL / track_slug

        if two_stems:
            v_src = model_out / "vocals.wav"
            i_src = model_out / "no_vocals.wav"
            if not v_src.exists() or not i_src.exists():
                raise FileNotFoundError(f"Demucs 2-stem output files missing in {model_out}")
            shutil.copy2(v_src, output_dir / f"{track_slug}_vocals.wav")
            shutil.copy2(i_src, output_dir / f"{track_slug}_instrumental.wav")
        else:
            for stem_name in ["vocals", "drums", "bass", "other"]:
                src = model_out / f"{stem_name}.wav"
                if src.exists():
                    shutil.copy2(src, output_dir / f"{track_slug}_{stem_name}.wav")

        shutil.rmtree(demucs_temp, ignore_errors=True)
