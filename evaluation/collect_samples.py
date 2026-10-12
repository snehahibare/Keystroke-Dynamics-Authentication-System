"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
Interactive Human Keystroke Sample Collection Helper

Allows real people to record typing samples via terminal or connects to the
FastAPI endpoint /evaluation/save-sample.
"""

import sys
import os
import time
import requests

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from database import (
    create_user,
    get_user_by_username,
    save_evaluation_sample,
    get_all_enrolled_users,
    init_db
)
from feature_extraction import extract_features

REFERENCE_SENTENCE = "the quick brown fox jumps over the lazy dog to verify cybersecurity patterns"

def interactive_collector():
    init_db()
    print("=" * 70)
    print("Keystroke Dynamics Evaluation Data Collector (Live Human Typists)")
    print("=" * 70)
    print(f"Standard Reference Sentence:\n\"{REFERENCE_SENTENCE}\"\n")

    enrolled = get_all_enrolled_users()
    enrolled_names = [u["username"] for u in enrolled]
    print(f"Currently Enrolled Profiles: {enrolled_names}")

    typist_name = input("Enter YOUR Name/Identifier (who is physically typing): ").strip().lower()
    if not typist_name:
        print("Typist name is required.")
        return

    target_profile = input(f"Enter TARGET Profile to test (e.g., {enrolled_names[0] if enrolled_names else 'alice'}): ").strip().lower()
    if not target_profile:
        print("Target profile is required.")
        return

    is_genuine = (typist_name == target_profile)
    print(f"\nLabel: {'GENUINE ATTEMPT' if is_genuine else 'IMPOSTOR ATTEMPT'}")
    print("\nInstructions: Press Enter to begin. Then type the reference sentence naturally and press Enter when done.\n")
    input("Press Enter when ready to start typing...")

    print(f"\n>>> TYPE NOW:\n{REFERENCE_SENTENCE}")

    t0 = time.time()
    typed_text = input(">>> ")
    t1 = time.time()

    duration = t1 - t0
    char_count = len(typed_text)
    if char_count < 15:
        print(f"Sample too short ({char_count} chars). Discarded.")
        return

    cpm = (char_count / duration) * 60.0
    print(f"\nCaptured {char_count} characters in {duration:.2f}s ({cpm:.1f} CPM)")
    print("For full microsecond per-key dwell and flight capturing, use the web UI at http://127.0.0.1:3000")

if __name__ == "__main__":
    interactive_collector()
