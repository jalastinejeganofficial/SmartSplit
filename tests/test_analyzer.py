"""
Unit tests for SmartSplit MusicAnalyzer.
"""

import sys
from pathlib import Path

# Add package root to sys.path
package_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(package_root))

from smartsplit.analyzer import MusicAnalyzer

def test_analyze():
    sample_mp3 = Path("c:/spotify_music_remover/Rathinamo.mp3")
    if not sample_mp3.exists():
        print("Sample MP3 not found, skipping.")
        return

    analyzer = MusicAnalyzer(sample_limit_seconds=30.0)
    result = analyzer.analyze(sample_mp3)

    print("=" * 60)
    print("SMARTSPLIT MUSIC ANALYZER TEST RESULT")
    print("=" * 60)
    print(f"File: {result['file_name']}")
    print(f"Duration: {result['audio_specs']['duration_formatted']} ({result['audio_specs']['duration_seconds']}s)")
    print(f"Tempo: {result['audio_specs']['estimated_bpm']} BPM")
    print(f"Dynamic Range: {result['audio_specs']['dynamic_range_db']} dB")
    print(f"Stereo Width: {result['mix_staging']['stereo_width']} (Mid: {result['mix_staging']['center_energy_pct']}%, Side: {result['mix_staging']['side_energy_pct']}%)")
    print("\n--- DETECTED INSTRUMENTS ---")
    for inst in result['detected_instruments']:
        status = "PRESENT" if inst['detected'] else "ABSENT"
        print(f"[{status}] {inst['name']:25} | Confidence: {inst['confidence']:5.1f}% | Placement: {inst['mix_placement']}")
        print(f"        Evidence: {inst['acoustic_evidence']}")
    
    print("\n--- FREQUENCY DISTRIBUTION ---")
    freqs = result['frequency_distribution']
    print(f"Sub-Bass: {freqs['sub_bass_pct']}% | Bass: {freqs['bass_pct']}% | Low-Mid: {freqs['low_mid_pct']}% | High-Mid: {freqs['high_mid_pct']}% | Air: {freqs['air_pct']}%")

    print("\n--- VOCAL SEPARABILITY ---")
    sep = result['vocal_separability']
    print(f"Rating: {sep['rating']} ({sep['removability_score_pct']}%)")
    print(f"Recommended: {sep['recommended_method']}")
    print(f"Notes: {sep['technical_details']}")
    print("\nSUMMARY:\n" + result['summary'])
    print("=" * 60)

    assert result['status'] == "success"
    assert len(result['detected_instruments']) > 0
    print("\n>>> ALL ANALYZER ASSERTIONS PASSED! <<<")

if __name__ == "__main__":
    test_analyze()
