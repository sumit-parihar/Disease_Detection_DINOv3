"""
source_alarm_thresholds.py  -  Batch-alarm thresholds for the DINOv3 system, from SOURCE data only,
with the rule of the extended study (capstone-paper-extended/src/explore_pp2020.py, part 3), fixed
before any S-test, PP2020 or PlantDoc output exists.

  S2 = total-variation distance between a batch's predicted-class shares and the S-calibration
       predicted-class shares (same seed). Threshold = 99th percentile over 2,000 random batches of
       reliability-training ORIGINAL images (no perturbed views), batch sizes 20 and 50.
  S1 = share of a batch beyond the 95th percentile of S-calibration centre distances
       (centre distance = ||(x - mean) / std|| / sqrt(d), mean/std from S-calibration features).
       Same threshold rule. Descriptive only.
Also reported: false-alarm rates on held-out reliability-validation originals.
Random draws: numpy default_rng(2026 + seed), batches drawn without replacement.

OUTPUT results/source_alarm_thresholds.csv
"""

import sys

import numpy as np
import pandas as pd

from common import RESULTS_DIR, SEEDS

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
BATCH_SIZES = (20, 50)
ALARM_QUANTILE = 99
OUT = RESULTS_DIR / "source_alarm_thresholds.csv"


def centre_distance(x, mean, std):
    return np.linalg.norm((x - mean) / std, axis=1) / np.sqrt(x.shape[1])


def batch_values(statistic, n_items, size, rng, draws=BOOTSTRAP_RESAMPLES):
    return np.array([statistic(rng.choice(n_items, size, replace=False)) for _ in range(draws)])


def tv_distance(predicted, reference_shares):
    shares = np.bincount(predicted, minlength=3) / len(predicted)
    return 0.5 * np.abs(shares - reference_shares).sum()


def main():
    if OUT.exists():
        sys.exit("Thresholds already exist.")
    rows = []
    for seed in SEEDS:
        cal = np.load(RESULTS_DIR / f"s_calibration_features_seed{seed}.npz")
        cal_x = cal["features"].astype(np.float64)
        mean, std = cal_x.mean(axis=0), cal_x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        q95 = np.percentile(centre_distance(cal_x, mean, std), 95)
        cal_shares = np.bincount(cal["logits"].argmax(axis=1), minlength=3) / len(cal_x)

        views = np.load(RESULTS_DIR / f"reliability_features_seed{seed}.npz")
        originals = views["view_types"] == "original"
        train = originals & (views["reliability_split"] == "reliability_train")
        val = originals & (views["reliability_split"] == "reliability_val")
        feats = views["features"].astype(np.float64)
        beyond_train, beyond_val = (centre_distance(feats[m], mean, std) > q95 for m in (train, val))
        pred_train, pred_val = views["predicted_label_index"][train], views["predicted_label_index"][val]

        rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
        for size in BATCH_SIZES:
            thr_s1 = np.percentile(batch_values(lambda i: beyond_train[i].mean(), len(beyond_train), size, rng), ALARM_QUANTILE)
            thr_s2 = np.percentile(batch_values(lambda i: tv_distance(pred_train[i], cal_shares), len(pred_train), size, rng), ALARM_QUANTILE)
            fa_s1 = (batch_values(lambda i: beyond_val[i].mean(), len(beyond_val), size, rng) > thr_s1).mean()
            fa_s2 = (batch_values(lambda i: tv_distance(pred_val[i], cal_shares), len(pred_val), size, rng) > thr_s2).mean()
            rows.append({"seed": seed, "batch_size": size, "S1_threshold": thr_s1, "S2_threshold": thr_s2,
                         "S1_false_alarm_rel_val": float(fa_s1), "S2_false_alarm_rel_val": float(fa_s2),
                         "calibration_p95_centre_distance": q95,
                         "calibration_predicted_shares": " ".join(f"{s:.4f}" for s in cal_shares)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False, lineterminator="\n")
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
