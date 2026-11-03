"""
run_all_tests.py
-----------------
Runs the full OmniVoice test suite in one go, in the same order as the
setup guide's Section 5. Loading the model once and reusing it here would
be faster, but each script is kept standalone (matching your Piper
project's structure) so any single test can be re-run in isolation.

Run:
    python run_all_tests.py
"""

import subprocess
import sys

SCRIPTS = [
    "dummy_llm_client.py",
    "check_latency.py",
    "edge_case_test.py",
    "test_voice_clone.py",
    "test_voice_design.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'=' * 70}\nRunning {script}\n{'=' * 70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n{script} exited with an error (code {result.returncode}). "
                  f"Continuing with the remaining scripts.")

    print("\nAll tests attempted. Check output/ for generated .wav files and latency_results.csv.")


if __name__ == "__main__":
    main()
