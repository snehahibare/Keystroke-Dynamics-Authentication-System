import sys
import os
import sqlite3
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from model import score_sample

conn = sqlite3.connect("backend/keystroke_dna.db")
eval_samples = conn.execute("SELECT * FROM evaluation_samples WHERE target_username = 'alice_fast'").fetchall()

print("Evaluating test samples for target profile 'alice_fast':")
for row in eval_samples:
    typist = row[1]
    is_gen = row[3]
    vec = json.loads(row[4])
    r_oc = score_sample(11, vec, "ocsvm")
    r_if = score_sample(11, vec, "isolation_forest")
    raw_oc = r_oc["raw_anomaly_score"]
    c_oc = r_oc["confidence_score"]
    raw_if = r_if["raw_anomaly_score"]
    c_if = r_if["confidence_score"]
    print(f"Typist: {typist:<14} | Gen: {is_gen} | OCSVM: raw={raw_oc:>7.4f}, conf={c_oc:>5.1f}% | IF: raw={raw_if:>7.4f}, conf={c_if:>5.1f}%")
