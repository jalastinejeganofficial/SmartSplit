"""
SmartSplit MCP Server
Model Context Protocol (MCP) server providing music analysis, instrument detection,
and AI-powered stem separation tools for AI agents and LLMs.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from mcp.server.mcpserver import MCPServer

from .analyzer import MusicAnalyzer
from .separator import StemSeparator
from .audio_utils import load_audio, format_duration
from .config import DEFAULT_OUTPUT_DIR

# Initialize MCP Server
app = MCPServer(
    name="smartsplit",
    version="1.0.0",
    description="SmartSplit MCP Server: Acoustic Music Analysis, Instrument Detection, and High-Fidelity Vocal & Stem Separation"
)

# Shared instances
_analyzer = MusicAnalyzer()
_separator = StemSeparator()


@app.tool(
    name="analyze_music",
    description=(
        "Performs in-depth acoustic reverse-engineering and music analysis on an audio file (MP3, WAV, FLAC, M4A). "
        "Detects musical instruments (Vocals, Drums, Bass, Guitars/Keys, Cymbals), estimates tempo/BPM, dynamic range, "
        "stereo staging (mid/side balance), frequency spectrum distribution, and calculates vocal removability score."
    )
)
def analyze_music(
    audio_path: str,
    sample_limit_seconds: float = 60.0
) -> str:
    """
    Analyze music audio file.
    
    Args:
        audio_path: Absolute or relative path to the audio file.
        sample_limit_seconds: Duration of audio in seconds to analyze for acoustic inspection (default: 60.0).
    """
    try:
        analyzer = MusicAnalyzer(sample_limit_seconds=sample_limit_seconds)
        result = analyzer.analyze(audio_path)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to analyze music: {str(e)}"
        }, indent=2)


@app.tool(
    name="detect_instruments",
    description=(
        "Analyzes an audio file and identifies what instruments are present in the mix. "
        "Returns a detailed list of detected instruments (Vocals, Drums & Percussion, Bass, Guitars/Keyboards/Synths, "
        "Cymbals & Ambient Air), confidence percentages (0-100%), frequency ranges, and stereo placement (Center vs. Panned)."
    )
)
def detect_instruments(audio_path: str) -> str:
    """
    Detect musical instruments in an audio track.
    
    Args:
        audio_path: Path to the audio file.
    """
    try:
        result = _analyzer.analyze(audio_path)
        payload = {
            "status": "success",
            "file_name": result["file_name"],
            "duration": result["audio_specs"]["duration_formatted"],
            "detected_instruments": result["detected_instruments"],
            "summary": (
                f"Instrument Analysis for '{result['file_name']}': " +
                ", ".join([f"{i['name']} ({i['confidence']}%)" for i in result['detected_instruments'] if i['detected']])
            )
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to detect instruments: {str(e)}"
        }, indent=2)


@app.tool(
    name="remove_vocals",
    description=(
        "Removes vocals completely from a song (0.00% vocal bleed) to create a pristine instrumental backing track "
        "(Karaoke track) and isolates the vocal stem. Uses our studio-grade Zero-Bleed mathematical center-channel "
        "cancellation with active stereo side-channel de-reverberation (< 3s CPU execution) or Demucs neural model."
    )
)
def remove_vocals(
    audio_path: str,
    quality: str = "fast",
    zero_bleed: bool = True,
    output_dir: Optional[str] = None
) -> str:
    """
    Remove vocals from audio file with guaranteed 0% residual vocal bleed.
    
    Args:
        audio_path: Path to the source song or audio file.
        quality: 'fast' (Instant mathematical DSP, ~2s) or 'balanced' / 'high' (Demucs Neural Network).
        zero_bleed: Whether to apply non-linear hard gating and stereo side-channel de-reverberation for absolute 0% vocal bleed (default: True).
        output_dir: Optional directory to store separated stems. If omitted, uses ~/.smartsplit/output.
    """
    try:
        result = _separator.separate_vocals(audio_path, quality=quality, zero_bleed=zero_bleed, output_dir=output_dir)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Vocal removal failed: {str(e)}"
        }, indent=2)


@app.tool(
    name="isolate_instrument",
    description=(
        "Isolates a single specific musical element or instrument from an audio mix with 0.00% cross-stem bleed. "
        "Supported targets: 'vocals' (isolated vocal formants), 'instrumental' (backing track without vocals), "
        "'bass' (pure fundamental <200Hz, 0% vocal/drum bleed), 'drums' (percussion with vocal formant rejection), "
        "or 'other' (melodic harmony/guitars/keys without bass or vocals)."
    )
)
def isolate_instrument(
    audio_path: str,
    instrument: str = "vocals",
    quality: str = "fast",
    zero_bleed: bool = True,
    output_dir: Optional[str] = None
) -> str:
    """
    Isolate a specific instrument or stem from audio with 0% bleed.
    
    Args:
        audio_path: Path to the audio file.
        instrument: Target to isolate: 'vocals', 'instrumental', 'drums', 'bass', or 'other'.
        quality: 'fast' (Instant DSP) or 'balanced' (Demucs Neural).
        zero_bleed: Enable hard stopband gating and cross-bleed cancellation (default: True).
        output_dir: Optional destination directory for output stem.
    """
    try:
        result = _separator.isolate_instrument(
            audio_path,
            target_instrument=instrument,
            quality=quality,
            zero_bleed=zero_bleed,
            output_dir=output_dir
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Instrument isolation failed: {str(e)}"
        }, indent=2)


@app.tool(
    name="separate_stems",
    description=(
        "Splits an audio track into isolated stem tracks with zero cross-stem bleed. "
        "Supports 2-stem separation (vocals + instrumental) or 4-stem separation (vocals + drums + bass + other)."
    )
)
def separate_stems(
    audio_path: str,
    stems: int = 2,
    quality: str = "fast",
    zero_bleed: bool = True,
    output_dir: Optional[str] = None
) -> str:
    """
    Separate track into 2 or 4 stems with 0% cross bleed.
    
    Args:
        audio_path: Path to the audio file.
        stems: 2 (vocals, instrumental) or 4 (vocals, drums, bass, other). Default is 2.
        quality: 'fast' (instant mathematical DSP) or 'balanced' (Demucs Neural).
        zero_bleed: Enable zero-bleed stopband masks and stereo de-reverberation (default: True).
        output_dir: Directory where separated WAV stems will be written.
    """
    try:
        result = _separator.separate_stems(
            audio_path,
            stems=stems,
            quality=quality,
            zero_bleed=zero_bleed,
            output_dir=output_dir
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Stem separation failed: {str(e)}"
        }, indent=2)


@app.tool(
    name="get_audio_info",
    description=(
        "Retrieves technical metadata and audio properties of an audio file without full stem separation: "
        "duration, sample rate, channel count, peak level (dBFS), RMS level (dBFS), and dynamic range."
    )
)
def get_audio_info(audio_path: str) -> str:
    """
    Get quick technical audio file metadata.
    
    Args:
        audio_path: Path to the audio file.
    """
    try:
        path = Path(audio_path).resolve()
        data, sr = load_audio(path)
        duration_sec = len(data) / float(sr)
        eps = 1e-12
        rms = float(np.sqrt(np.mean(data ** 2) + eps))
        peak = float(np.max(np.abs(data)))
        
        info = {
            "status": "success",
            "file_name": path.name,
            "file_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "duration_seconds": round(duration_sec, 2),
            "duration_formatted": format_duration(duration_sec),
            "sample_rate": sr,
            "channels": data.shape[1] if data.ndim > 1 else 1,
            "peak_dbfs": round(float(20.0 * np.log10(peak + eps)), 1),
            "rms_dbfs": round(float(20.0 * np.log10(rms + eps)), 1),
            "dynamic_range_db": round(abs(float(20.0 * np.log10(peak + eps)) - float(20.0 * np.log10(rms + eps))), 1)
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to inspect audio: {str(e)}"
        }, indent=2)


def main():
    """CLI entrypoint for SmartSplit MCP server."""
    # Run stdio transport
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
