"""
End-to-End MCP Server Tools Verification.
Tests tool functions exposed over Model Context Protocol (MCP).
"""

import sys
import json
from pathlib import Path

package_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(package_root))

from smartsplit.server import (
    analyze_music,
    detect_instruments,
    remove_vocals,
    isolate_instrument,
    get_audio_info,
    separate_stems
)

def run_tests():
    sample_mp3 = "c:/spotify_music_remover/Rathinamo.mp3"
    assert Path(sample_mp3).exists(), "Sample audio file missing!"
    out_dir = str(package_root / "tests" / "mcp_test_output")

    print("\n" + "=" * 65)
    print(" 1. TESTING MCP TOOL: get_audio_info")
    print("=" * 65)
    info_json = get_audio_info(sample_mp3)
    info = json.loads(info_json)
    print(f"File Name: {info['file_name']}")
    print(f"Duration: {info['duration_formatted']} ({info['duration_seconds']}s)")
    print(f"Channels: {info['channels']}, Sample Rate: {info['sample_rate']}Hz")
    print(f"Peak: {info['peak_dbfs']} dBFS, RMS: {info['rms_dbfs']} dBFS")
    assert info["status"] == "success"

    print("\n" + "=" * 65)
    print(" 2. TESTING MCP TOOL: detect_instruments")
    print("=" * 65)
    inst_json = detect_instruments(sample_mp3)
    inst_data = json.loads(inst_json)
    print(f"Summary: {inst_data['summary']}")
    for inst in inst_data['detected_instruments']:
        print(f"  • {inst['name']:25}: {inst['confidence']:5.1f}% [{inst['prominence']}]")
    assert inst_data["status"] == "success"
    assert len(inst_data["detected_instruments"]) >= 4

    print("\n" + "=" * 65)
    print(" 3. TESTING MCP TOOL: analyze_music")
    print("=" * 65)
    analysis_json = analyze_music(sample_mp3, sample_limit_seconds=30.0)
    analysis = json.loads(analysis_json)
    print(f"Estimated BPM: {analysis['audio_specs']['estimated_bpm']}")
    print(f"Vocal Removability: {analysis['vocal_separability']['removability_score_pct']}% ({analysis['vocal_separability']['rating']})")
    print(f"Recommended Separation: {analysis['vocal_separability']['recommended_method']}")
    assert analysis["status"] == "success"

    print("\n" + "=" * 65)
    print(" 4. TESTING MCP TOOL: remove_vocals (Fast Mathematical DSP)")
    print("=" * 65)
    vocal_json = remove_vocals(sample_mp3, quality="fast", output_dir=out_dir)
    vocal_res = json.loads(vocal_json)
    print(f"Status: {vocal_res['status']}")
    print(f"Processing Time: {vocal_res['processing_time_seconds']}s")
    print(f"Instrumental Path: {vocal_res['stems']['instrumental']}")
    print(f"Vocals Path: {vocal_res['stems']['vocals']}")
    assert vocal_res["status"] == "success"
    assert Path(vocal_res["stems"]["instrumental"]).exists()
    assert Path(vocal_res["stems"]["vocals"]).exists()

    print("\n" + "=" * 65)
    print(" 5. TESTING MCP TOOL: isolate_instrument (Bass)")
    print("=" * 65)
    bass_json = isolate_instrument(sample_mp3, instrument="bass", quality="fast", output_dir=out_dir)
    bass_res = json.loads(bass_json)
    print(f"Isolated: {bass_res['isolated_instrument']}")
    print(f"Path: {bass_res['file_path']}")
    print(f"Time: {bass_res['processing_time_seconds']}s")
    assert bass_res["status"] == "success"
    assert Path(bass_res["file_path"]).exists()

    print("\n" + "=" * 65)
    print(" >>> ALL MCP SERVER TOOLS VERIFIED SUCCESSFULLY! <<< ")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
