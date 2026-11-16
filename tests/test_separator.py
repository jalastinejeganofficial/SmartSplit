"""
Unit tests for SmartSplit StemSeparator.
"""

import sys
from pathlib import Path

package_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(package_root))

from smartsplit.separator import StemSeparator

def test_separate():
    sample_mp3 = Path("c:/spotify_music_remover/Rathinamo.mp3")
    if not sample_mp3.exists():
        print("Sample MP3 not found, skipping.")
        return

    output_dir = package_root / "tests" / "test_output"
    separator = StemSeparator(default_output_dir=output_dir)

    print("=" * 60)
    print("TESTING FAST MATHEMATICAL VOCAL SEPARATION (<3s)")
    print("=" * 60)
    result = separator.separate_vocals(sample_mp3, quality="fast", output_dir=output_dir)

    print(f"Status: {result['status']}")
    print(f"Processing Time: {result['processing_time_seconds']}s")
    print(f"Algorithm: {result['algorithm_used']}")
    print(f"Instrumental Stem: {result['stems']['instrumental']}")
    print(f"Vocal Stem: {result['stems']['vocals']}")
    print(f"Summary: {result['summary']}")

    inst_file = Path(result['stems']['instrumental'])
    vocal_file = Path(result['stems']['vocals'])
    assert inst_file.exists(), "Instrumental WAV file was not created!"
    assert vocal_file.exists(), "Vocal WAV file was not created!"
    assert inst_file.stat().st_size > 1000, "Instrumental file is too small/empty"
    assert vocal_file.stat().st_size > 1000, "Vocal file is too small/empty"

    print("\n=" * 60)
    print("TESTING INSTRUMENT ISOLATION (BASS)")
    print("=" * 60)
    iso_result = separator.isolate_instrument(sample_mp3, target_instrument="bass", quality="fast", output_dir=output_dir)
    print(f"Isolated: {iso_result['isolated_instrument']}")
    print(f"Path: {iso_result['file_path']}")
    print(f"Time: {iso_result['processing_time_seconds']}s")
    assert Path(iso_result['file_path']).exists()

    print("\n>>> ALL SEPARATOR ASSERTIONS PASSED! <<<")

if __name__ == "__main__":
    test_separate()
