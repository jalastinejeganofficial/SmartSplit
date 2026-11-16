"""
SmartSplit MCP - AI-Powered Music Analysis & Stem Separation Server
Model Context Protocol (MCP) tool suite enabling AI agents and LLMs to analyze
audio mixes, detect musical instruments, and separate vocals and stems.
"""

__version__ = "1.0.0"
__author__ = "SmartSplit Team"

from .analyzer import MusicAnalyzer, AudioAnalysisResult
from .separator import StemSeparator, SeparationResult
from .dsp_math import fast_spectral_separation

__all__ = [
    "MusicAnalyzer",
    "AudioAnalysisResult",
    "StemSeparator",
    "SeparationResult",
    "fast_spectral_separation",
]
