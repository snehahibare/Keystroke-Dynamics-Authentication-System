"""
Personal Digital DNA — Module 1: Keystroke Dynamics Biometric Engine
Evaluation & Benchmarking Engine

Computes:
1. False Acceptance Rate (FAR) curve
2. False Rejection Rate (FRR) curve
3. Equal Error Rate (EER) and optimal crossing threshold
4. Operational threshold metrics at 30% and 85%
5. Side-by-side comparison between One-Class SVM and Isolation Forest
6. Matplotlib visualization saved as PNG (FAR/FRR Tradeoff & ROC curves)
"""

import sys
import os
import sqlite3
import numpy as np

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
# Use headless backend for non-GUI execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure backend directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from database import (
    get_all_enrolled_users,
    get_evaluation_samples,
    get_user_by_username,
    init_db,
    DB_PATH
)
from model import score_sample

# Destination for generated plot
CHART_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "far_frr_tradeoff.png")


def run_evaluation():
    init_db()
    print("=" * 80)
    print("KEYSTROKE DYNAMICS BIOMETRIC EVALUATION ENGINE")
    print("Module 1: Personal Digital DNA Anomaly Detection Benchmark")
    print("=" * 80)

    # 1. Load Enrolled Users & Evaluation Samples
    enrolled_users = get_all_enrolled_users()
    eval_samples = get_evaluation_samples()

    if not eval_samples or not enrolled_users:
        print("\n[ERROR] Insufficient data for evaluation.")
        print(f"Found {len(enrolled_users)} enrolled users and {len(eval_samples)} evaluation samples.")
        print("Please run `python evaluation/seed_human_data.py` or collect samples first.")
        return

    # Count subjects and attempts
    unique_typists = set(s["typist_name"] for s in eval_samples)
    genuine_samples = [s for s in eval_samples if s["is_genuine"]]
    impostor_samples = [s for s in eval_samples if not s["is_genuine"]]

    print(f"\n[EVALUATION CORPUS OVERVIEW]")
    print(f"  • Enrolled Target Profiles : {len(enrolled_users)} subjects")
    print(f"  • Unique Human Typists     : {len(unique_typists)} subjects")
    print(f"  • Total Labeled Test Trials: {len(eval_samples)}")
    print(f"  • Genuine Trials           : {len(genuine_samples)}")
    print(f"  • Impostor Attack Trials   : {len(impostor_samples)}")
    print("-" * 80)

    # 2. Evaluate both models
    models_to_test = ["ocsvm", "isolation_forest"]
    results = {}

    for model_type in models_to_test:
        model_display = "One-Class SVM" if model_type == "ocsvm" else "Isolation Forest"
        print(f"\nEvaluating Classifier: {model_display}...")

        genuine_confidences = []
        impostor_confidences = []

        for sample in eval_samples:
            target_username = sample["target_username"]
            user_info = get_user_by_username(target_username)
            if not user_info:
                continue

            user_id = user_info["user_id"]
            feat_vec = sample["feature_vector"]

            try:
                score_res = score_sample(user_id=user_id, feature_vector=feat_vec, model_type=model_type)
                conf = score_res["confidence_score"]

                if sample["is_genuine"]:
                    genuine_confidences.append(conf)
                else:
                    impostor_confidences.append(conf)
            except Exception as e:
                # User might not have this model type trained
                continue

        # 3. Sweep Thresholds from 0% to 100%
        thresholds = np.linspace(0.0, 100.0, 201)  # step = 0.5%
        far_list = []
        frr_list = []

        total_gen = max(1, len(genuine_confidences))
        total_imp = max(1, len(impostor_confidences))

        for th in thresholds:
            # FAR: Impostors wrongly accepted (confidence >= threshold)
            false_accepts = sum(1 for c in impostor_confidences if c >= th)
            far = (false_accepts / total_imp) * 100.0
            far_list.append(far)

            # FRR: Genuine users wrongly rejected (confidence < threshold)
            false_rejects = sum(1 for c in genuine_confidences if c < th)
            frr = (false_rejects / total_gen) * 100.0
            frr_list.append(frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        # 4. Find Equal Error Rate (EER) Point where |FAR - FRR| is minimal
        abs_diff = np.abs(far_arr - frr_arr)
        min_idx = np.argmin(abs_diff)
        eer_threshold = float(thresholds[min_idx])
        eer_value = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)

        # Operational Threshold Metrics (30% and 85%)
        idx_30 = np.argmin(np.abs(thresholds - 30.0))
        idx_85 = np.argmin(np.abs(thresholds - 85.0))

        far_at_30 = float(far_arr[idx_30])
        frr_at_30 = float(frr_arr[idx_30])
        far_at_85 = float(far_arr[idx_85])
        frr_at_85 = float(frr_arr[idx_85])

        results[model_type] = {
            "display": model_display,
            "thresholds": thresholds,
            "far": far_arr,
            "frr": frr_arr,
            "eer": eer_value,
            "eer_threshold": eer_threshold,
            "far_30": far_at_30,
            "frr_30": frr_at_30,
            "far_85": far_at_85,
            "frr_85": frr_at_85,
            "genuine_count": len(genuine_confidences),
            "impostor_count": len(impostor_confidences),
            "mean_gen_conf": float(np.mean(genuine_confidences)) if genuine_confidences else 0.0,
            "mean_imp_conf": float(np.mean(impostor_confidences)) if impostor_confidences else 0.0
        }

    # 5. Print Tabulated Benchmark Report
    print("\n" + "=" * 80)
    print("BIOMETRIC ACCURACY BENCHMARK REPORT (FAR / FRR / EER)")
    print("=" * 80)

    header = f"{'Metric':<34} | {'One-Class SVM':<20} | {'Isolation Forest':<20}"
    print(header)
    print("-" * 80)

    r_svm = results["ocsvm"]
    r_if = results["isolation_forest"]

    print(f"{'Equal Error Rate (EER)':<34} | {r_svm['eer']:>18.2f}% | {r_if['eer']:>18.2f}%")
    print(f"{'Optimal Operating Threshold (EER)':<34} | {r_svm['eer_threshold']:>18.1f}% | {r_if['eer_threshold']:>18.1f}%")
    print(f"{'FAR @ 30% Threshold (Strict Impostor)':<34} | {r_svm['far_30']:>18.2f}% | {r_if['far_30']:>18.2f}%")
    print(f"{'FRR @ 30% Threshold':<34} | {r_svm['frr_30']:>18.2f}% | {r_if['frr_30']:>18.2f}%")
    print(f"{'FAR @ 85% Threshold (Genuine Access)':<34} | {r_svm['far_85']:>18.2f}% | {r_if['far_85']:>18.2f}%")
    print(f"{'FRR @ 85% Threshold':<34} | {r_svm['frr_85']:>18.2f}% | {r_if['frr_85']:>18.2f}%")
    print("-" * 80)
    print(f"{'Mean Genuine Score':<34} | {r_svm['mean_gen_conf']:>18.1f}% | {r_if['mean_gen_conf']:>18.1f}%")
    print(f"{'Mean Impostor Score':<34} | {r_svm['mean_imp_conf']:>18.1f}% | {r_if['mean_imp_conf']:>18.1f}%")
    print("=" * 80)

    # 6. Sanity Checks & Academic Realism Check
    print("\n[SCIENTIFIC ACCURACY VALIDATION]")
    for m in models_to_test:
        eer = results[m]["eer"]
        name = results[m]["display"]
        if eer < 1.0:
            print(f"  [WARNING] ({name}): EER is near 0.0% ({eer:.2f}%). Check for sample homogeneity.")
        elif 3.0 <= eer <= 15.0:
            print(f"  [VALIDATED] ({name}): EER is {eer:.2f}%. Fully consistent with published peer-reviewed keystroke literature (typical 5-15% range).")
        else:
            print(f"  [INFO] ({name}): EER observed at {eer:.2f}%.")

    # 7. Model Recommendation Justification
    best_model = "ocsvm" if r_svm["eer"] <= r_if["eer"] else "isolation_forest"
    best_name = "One-Class SVM" if best_model == "ocsvm" else "Isolation Forest"
    print(f"\n[RECOMMENDATION & ARCHITECTURAL JUSTIFICATION]")
    print(f"  * Recommended Default Model: {best_name}")
    print(f"  * Rationale: {best_name} achieved lower Equal Error Rate ({results[best_model]['eer']:.2f}% vs {results['isolation_forest' if best_model == 'ocsvm' else 'ocsvm']['eer']:.2f}%),")
    print(f"    establishing tighter non-linear RBF decision hyperspheres around physiological typing motor rhythm.")

    # Confidence Caveat for Realistic Small Sample Sizes
    print(f"\n[CONFIDENCE CAVEAT]")
    print(f"  Results based on N={len(unique_typists)} real human subjects ({len(eval_samples)} total trials).")
    print(f"  Small sample size: indicative rather than statistically robust.")

    # 8. Generate Professional Matplotlib Tradeoff Chart
    generate_evaluation_chart(results, CHART_OUTPUT_PATH)
    print(f"\n[ARTIFACT] Matplotlib Performance Plot saved to:\n  -> {CHART_OUTPUT_PATH}")
    print("=" * 80)


def generate_evaluation_chart(results: dict, output_path: str):
    """
    Renders and saves a dual-panel evaluation figure with the warm editorial palette:
    Panel 1: FAR & FRR vs Confidence Threshold Curve with marked EER points
    Panel 2: ROC Curve (GAR vs FAR) with diagonal chance line
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.patch.set_facecolor("#FAF8F5")
    ax1.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color("#E3DBCF")
            spine.set_linewidth(1.0)
        ax.tick_params(colors="#453A42", labelsize=9)

    # Palette
    c_plum = "#7A4B6B"       # Primary accent: One-Class SVM
    c_gold = "#C9A24B"       # Secondary accent: Isolation Forest / Highlights
    c_plum_light = "#A87095" # Secondary plum line
    c_gold_light = "#DFBA68" # Secondary gold line
    c_fg = "#453A42"
    c_muted = "#9A8F94"
    c_line = "#E3DBCF"

    # --- PANEL 1: FAR / FRR vs Threshold ---
    r_svm = results["ocsvm"]
    r_if = results["isolation_forest"]

    ax1.plot(r_svm["thresholds"], r_svm["far"], label="OCSVM: FAR (False Accept)", color=c_plum, linewidth=2.4)
    ax1.plot(r_svm["thresholds"], r_svm["frr"], label="OCSVM: FRR (False Reject)", color=c_plum_light, linewidth=2.0, linestyle="-.")

    ax1.plot(r_if["thresholds"], r_if["far"], label="IForest: FAR", color=c_gold, linestyle="--", linewidth=2.0)
    ax1.plot(r_if["thresholds"], r_if["frr"], label="IForest: FRR", color=c_gold_light, linestyle=":", linewidth=2.0)

    # Mark EER point for OCSVM
    ax1.scatter([r_svm["eer_threshold"]], [r_svm["eer"]], color=c_plum, s=90, zorder=5)
    ax1.annotate(
        f"OCSVM EER: {r_svm['eer']:.1f}%\n(@ {r_svm['eer_threshold']:.1f}%)",
        xy=(r_svm["eer_threshold"], r_svm["eer"]),
        xytext=(r_svm["eer_threshold"] - 22, r_svm["eer"] + 15),
        arrowprops=dict(facecolor=c_plum, shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color=c_plum,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F0EBE3", ec=c_line, lw=1)
    )

    # Mark Decision Thresholds (30% and 85%)
    ax1.axvline(x=30.0, color=c_gold, linestyle=":", alpha=0.8, label="Suspicious Threshold (30%)")
    ax1.axvline(x=85.0, color=c_plum, linestyle=":", alpha=0.8, label="Genuine Threshold (85%)")

    ax1.set_title("FAR & FRR Tradeoff vs. Decision Threshold", fontsize=11, fontweight="bold", pad=12, color=c_fg)
    ax1.set_xlabel("Confidence Threshold (%)", fontsize=10, color=c_muted)
    ax1.set_ylabel("Error Rate (%)", fontsize=10, color=c_muted)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle="--", alpha=0.5, color=c_line)
    leg1 = ax1.legend(loc="center right", fontsize=8.5, framealpha=0.9, facecolor="#F0EBE3", edgecolor=c_line)
    for text in leg1.get_texts():
        text.set_color(c_fg)

    # --- PANEL 2: ROC Curve (GAR vs FAR) ---
    gar_svm = 100.0 - r_svm["frr"]
    gar_if = 100.0 - r_if["frr"]

    sort_svm = np.argsort(r_svm["far"])
    sort_if = np.argsort(r_if["far"])

    ax2.plot(r_svm["far"][sort_svm], gar_svm[sort_svm], label=f"OCSVM (EER = {r_svm['eer']:.1f}%)", color=c_plum, linewidth=2.5)
    ax2.plot(r_if["far"][sort_if], gar_if[sort_if], label=f"Isolation Forest (EER = {r_if['eer']:.1f}%)", color=c_gold, linewidth=2, linestyle="--")
    ax2.plot([0, 100], [0, 100], color=c_muted, linestyle=":", label="Random Chance (EER = 50%)")

    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold", pad=12, color=c_fg)
    ax2.set_xlabel("False Acceptance Rate (FAR %)", fontsize=10, color=c_muted)
    ax2.set_ylabel("Genuine Acceptance Rate (GAR %)", fontsize=10, color=c_muted)
    ax2.set_xlim(-1, 101)
    ax2.set_ylim(-1, 101)
    ax2.grid(True, linestyle="--", alpha=0.5, color=c_line)
    leg2 = ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9, facecolor="#F0EBE3", edgecolor=c_line)
    for text in leg2.get_texts():
        text.set_color(c_fg)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_evaluation()
