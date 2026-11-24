"""
evaluate.py  -  Step 9b of the plan: the ONE-TIME evaluation of the frozen DINOv3 system on S-test,
PP2020 and PlantDoc, exactly as pre-registered (prereg/dinov3_preregistration.md). Labels are joined
here for the first time.

Prechecks (any failure stops): outputs do not exist; the pre-registration is the sealed version; the
freeze record verifies; every prediction file matches the hash recorded right after inference; the
prediction files are committed and pushed.

Metrics come from the ORIGINAL metrics.py (AURC with tie averaging, AUROC/AUPRC-error, selective risk,
classification and calibration summaries). Statistics as in the extended study: 2,000 resamples of
whole clusters, seed 2026, the same resamples for all methods and seeds, seeds resampled jointly.

RUN once:                      venv/Scripts/python.exe src/evaluate.py
CODE TEST (random labels):     venv/Scripts/python.exe src/evaluate.py --dry-run-random-labels <folder>
OUTPUT results/evaluation/
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXTENDED_ROOT, FROZEN_ROOT, RESULTS_DIR, ROOT, SEEDS, sha256, use_frozen_code

use_frozen_code()

from scipy.stats import spearmanr   # noqa: E402
from sklearn.metrics import precision_recall_fscore_support   # noqa: E402

from metrics import (auprc_error, aurc, auroc_error, calibration_summary,   # noqa: E402  (original)
                     classification_summary, risk_coverage_curve, selective_risk_at)

PREREG = ROOT / "prereg" / "dinov3_preregistration.md"
PREREG_SHA = "2e6b58847aad4ea1aef2344cc1759078108d36f33721d9f200be42391f72a9cf"
CLASSES = ["healthy", "scab", "rust"]
DATASETS = ("s_test", "pp2020", "plantdoc")
TARGETS = ("pp2020", "plantdoc")
INF = RESULTS_DIR / "inference"
HASHES = INF / "inference_hashes.txt"
OUT = RESULTS_DIR / "evaluation"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
MIN_ERRORS_PER_SEED = 50
COLLAPSE_THRESHOLD = 0.90
FIXED_COVERAGES = (0.5, 0.7, 0.8, 0.9)
METHODS = {"B1 MSP": "msp_score", "B2 Entropy": "entropy_score", "B3 Margin": "margin_score",
           "B4 Temp-scaled MSP": "temp_scaled_msp_score", "B5 Logits-only head": "head_logits_only",
           "A1 Features-only head": "head_features_only", "P1 Proposed head": "head_proposed",
           "E Energy": "energy_score"}
SETTINGS = json.loads((RESULTS_DIR / "frozen_calibration_settings.json").read_text(encoding="utf-8"))


def predictions_path(dataset, seed):
    return INF / f"{dataset}_predictions_seed{seed}.csv"


def labels_for(dataset):
    if dataset == "s_test":
        split = pd.read_csv(FROZEN_ROOT / "splits" / "split_assignments.csv", keep_default_na=False)
        return split.set_index("path")["label"]
    if dataset == "pp2020":
        manifest = pd.read_csv(FROZEN_ROOT / "manifests" / "manifest.csv", keep_default_na=False)
        return manifest[manifest["dataset"] == "target"].set_index("path")["label"]
    eligible = pd.read_csv(EXTENDED_ROOT / "manifests" / "plantdoc_eligible.csv", keep_default_na=False)
    return eligible.set_index("staged_path")["label"]


def precheck():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit("Evaluation outputs already exist. The evaluation is run once.")
    if sha256(PREREG) != PREREG_SHA:
        sys.exit("The pre-registration differs from the sealed version.")
    from freeze import verify_frozen
    verify_frozen()
    recorded = {}
    for line in HASHES.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            recorded[parts[1]] = parts[0]
    files = [predictions_path(d, s) for d in DATASETS for s in SEEDS]
    for path in files:
        if recorded.get(path.relative_to(ROOT).as_posix()) != sha256(path):
            sys.exit(f"{path.name} differs from the hash recorded after inference.")
    tracked = [p.relative_to(ROOT).as_posix() for p in files] + [HASHES.relative_to(ROOT).as_posix(),
                                                                 PREREG.relative_to(ROOT).as_posix()]
    status = subprocess.run(["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, capture_output=True, text=True)
    ahead = subprocess.run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=ROOT, capture_output=True, text=True)
    if status.returncode or status.stdout.strip() or ahead.returncode or ahead.stdout.strip() != "0":
        sys.exit("Prediction files are not committed and pushed. Push them first.")
    print("Prechecks passed: pre-registration, freeze, prediction hashes, commit and push.")


def cluster_resamples(cluster_ids):
    groups = [np.flatnonzero(cluster_ids == c) for c in pd.unique(cluster_ids)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            for _ in range(BOOTSTRAP_RESAMPLES)]


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if len(values) else (math.nan, math.nan)


def risk_at_coverage(errors, scores, coverage):
    return float(risk_coverage_curve(errors, scores)[1][math.ceil(coverage * len(errors)) - 1])


def tv_distance(predicted, reference_shares):
    return 0.5 * np.abs(np.bincount(predicted, minlength=3) / len(predicted) - reference_shares).sum()


def centre_distance(x, mean, std):
    return np.linalg.norm((x - mean) / std, axis=1) / np.sqrt(x.shape[1])


def safe(function, e, x):
    return function(e, x) if 0 < e.sum() < len(e) else math.nan


def spearman_gap(b5, en, mg):
    return spearmanr(b5, en)[0] - spearmanr(b5, mg)[0]


def load_tables(dataset, labels):
    tables = {}
    for seed in SEEDS:
        t = pd.read_csv(predictions_path(dataset, seed), keep_default_na=False)
        t["true_index"] = [CLASSES.index(labels[p]) for p in t["path"]]
        t["error"] = (t["predicted_index"] != t["true_index"]).astype(int)
        s = np.sort(t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64), axis=1)
        t["neg_logit_margin"] = -(s[:, 2] - s[:, 1])
        tables[seed] = t
    first = tables[SEEDS[0]]["path"].tolist()
    if any(t["path"].tolist() != first for t in tables.values()):
        sys.exit(f"{dataset}: prediction rows differ between seeds.")
    return tables


def evaluate_dataset(dataset, tables, alarm_thresholds):
    clusters = tables[SEEDS[0]]["cluster_id"].to_numpy()
    resamples = cluster_resamples(clusters)
    out = {"classification": [], "confusion": [], "calibration": [], "reliability": [], "seed_mean": [],
           "mechanism": [], "regime": [], "alarm": [], "decomposition": [], "magnitude": []}

    for seed, t in tables.items():
        y, p = t["true_index"].to_numpy(), t["predicted_index"].to_numpy()
        summary = classification_summary(y, p, CLASSES)
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        out["classification"].append({
            "dataset": dataset, "seed": seed, "images": len(t), "errors": int(t["error"].sum()),
            "error_rate": t["error"].mean(), "accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"],
            "underpowered": bool(t["error"].sum() < MIN_ERRORS_PER_SEED),
            **{f"precision_{c}": pr[i] for i, c in enumerate(CLASSES)},
            **{f"recall_{c}": rc[i] for i, c in enumerate(CLASSES)}, **{f"f1_{c}": f1[i] for i, c in enumerate(CLASSES)}})
        for i, c in enumerate(CLASSES):
            out["confusion"].append({"dataset": dataset, "seed": seed, "true": c,
                                     **{f"pred_{d}": int(summary["confusion_matrix"][i, j]) for j, d in enumerate(CLASSES)}})
        logits = t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
        before = calibration_summary(logits, y)
        after = calibration_summary(logits, y, SETTINGS[str(seed)]["temperature"])
        out["calibration"].append({"dataset": dataset, "seed": seed, "temperature": SETTINGS[str(seed)]["temperature"],
                                   **{f"{k}_before": v for k, v in before.items()},
                                   **{f"{k}_after": v for k, v in after.items()}})
        shares = np.bincount(p, minlength=3) / len(p)
        out["regime"].append({"dataset": dataset, "seed": seed, "top_class": CLASSES[int(shares.argmax())],
                              "top_class_share": float(shares.max()),
                              **{f"share_{c}": shares[i] for i, c in enumerate(CLASSES)}})

    # reliability with shared resamples; seed mean resampled jointly
    deltas = {m: np.zeros((len(SEEDS), BOOTSTRAP_RESAMPLES)) for m in METHODS}
    point = {m: [] for m in METHODS}
    for k, (seed, t) in enumerate(tables.items()):
        e, msp = t["error"].to_numpy(), t["msp_score"].to_numpy()
        for m, col in METHODS.items():
            x = t[col].to_numpy()
            d = aurc(e, x) - aurc(e, msp) if e.sum() > 0 else math.nan
            point[m].append(d)
            deltas[m][k] = [aurc(e[i], x[i]) - aurc(e[i], msp[i]) if e[i].sum() > 0 else math.nan for i in resamples]
            lo, hi = interval(deltas[m][k])
            row = {"dataset": dataset, "seed": seed, "method": m, "aurc": aurc(e, x), "delta_aurc_vs_msp": d,
                   "ci_low": lo, "ci_high": hi, "auroc_error": safe(auroc_error, e, x), "auprc_error": safe(auprc_error, e, x)}
            if m in SETTINGS[str(seed)]["thresholds"]:
                for tag in ("tau80", "tau90"):
                    sr = selective_risk_at(e, x, SETTINGS[str(seed)]["thresholds"][m][tag])
                    row[f"coverage_{tag}"], row[f"risk_{tag}"] = sr["coverage"], sr["selective_risk"]
            for c in FIXED_COVERAGES:
                row[f"risk_at_{int(c * 100)}pct"] = risk_at_coverage(e, x, c)
            out["reliability"].append(row)
    reliability = pd.DataFrame(out["reliability"])
    for m in METHODS:
        lo, hi = interval(np.nanmean(deltas[m], axis=0)) if np.isfinite(deltas[m]).any() else (math.nan, math.nan)
        sub = reliability[reliability["method"] == m]
        out["seed_mean"].append({"dataset": dataset, "method": m, "mean_aurc": sub["aurc"].mean(),
                                 "mean_delta_aurc_vs_msp": float(np.mean(point[m])), "ci_low": lo, "ci_high": hi,
                                 "mean_auroc_error": sub["auroc_error"].mean(), "mean_auprc_error": sub["auprc_error"].mean()})

    # mechanism: rho(B5, E) - rho(B5, -margin)
    gaps, draws = [], np.zeros((len(SEEDS), BOOTSTRAP_RESAMPLES))
    for k, t in enumerate(tables.values()):
        b5, en, mg = (t[c].to_numpy() for c in ("head_logits_only", "energy_score", "neg_logit_margin"))
        gaps.append(spearman_gap(b5, en, mg))
        draws[k] = [spearman_gap(b5[i], en[i], mg[i]) for i in resamples]
    lo, hi = interval(draws.mean(axis=0))
    out["mechanism"].append({"dataset": dataset, "seed_mean_gap": float(np.mean(gaps)), "ci_low": lo, "ci_high": hi,
                             **{f"gap_seed{s}": g for s, g in zip(SEEDS, gaps)}})

    # batch alarm (S2 confirmatory for the targets; S1 descriptive)
    for seed, t in tables.items():
        cal = np.load(RESULTS_DIR / f"s_calibration_features_seed{seed}.npz")
        cal_x = cal["features"].astype(np.float64)
        mean, std = cal_x.mean(axis=0), cal_x.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        q95 = np.percentile(centre_distance(cal_x, mean, std), 95)
        cal_shares = np.bincount(cal["logits"].argmax(axis=1), minlength=3) / len(cal_x)
        feats = np.load(INF / f"{dataset}_features_seed{seed}.npz")
        if feats["paths"].tolist() != t["path"].tolist():
            sys.exit(f"{dataset}: feature rows not aligned.")
        beyond = centre_distance(feats["features"].astype(np.float64), mean, std) > q95
        predicted = t["predicted_index"].to_numpy()
        rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
        for size in (20, 50):
            thr = alarm_thresholds[(seed, size)]
            picks = [rng.choice(len(t), size, replace=False) for _ in range(BOOTSTRAP_RESAMPLES)]
            s2 = np.array([tv_distance(predicted[i], cal_shares) for i in picks])
            s1 = np.array([beyond[i].mean() for i in picks])
            out["alarm"].append({"dataset": dataset, "seed": seed, "batch_size": size,
                                 "S2_threshold": thr["S2_threshold"], "S2_alarm_rate": float((s2 > thr["S2_threshold"]).mean()),
                                 "S2_median": float(np.median(s2)), "S1_threshold": thr["S1_threshold"],
                                 "S1_alarm_rate_descriptive": float((s1 > thr["S1_threshold"]).mean()),
                                 "share_beyond_p95": float(beyond.mean())})

    # exploratory: between/within-class decomposition and logit magnitude
    for seed, t in tables.items():
        e = t["error"].to_numpy()
        logits = t[["logit_healthy", "logit_scab", "logit_rust"]].to_numpy(dtype=np.float64)
        for m in ("B1 MSP", "B5 Logits-only head", "E Energy", "P1 Proposed head"):
            col = METHODS[m]
            x = t[col].to_numpy()
            class_mean = t.groupby("predicted_index")[col].transform("mean").to_numpy()
            within = []
            for k in range(3):
                mask = t["predicted_index"].to_numpy() == k
                if e[mask].sum() >= 10 and (1 - e[mask]).sum() >= 10:
                    within.append((CLASSES[k], auroc_error(e[mask], x[mask]), int(mask.sum())))
            out["decomposition"].append({
                "dataset": dataset, "seed": seed, "score": m, "auroc_overall": safe(auroc_error, e, x),
                "auroc_between_classes": safe(auroc_error, e, class_mean),
                "auroc_within_weighted": (sum(a * n for _, a, n in within) / sum(n for *_, n in within)) if within else math.nan,
                "within_detail": "; ".join(f"{c}:{a:.3f}(n={n})" for c, a, n in within)})
        lse = -t["energy_score"].to_numpy()
        for label, mask in (("correct", e == 0), ("wrong", e == 1)):
            out["magnitude"].append({"dataset": dataset, "seed": seed, "group": label, "images": int(mask.sum()),
                                     "median_logsumexp": float(np.median(lse[mask])) if mask.any() else math.nan})
    return out


def verdicts(frames):
    seed_mean = frames["seed_mean"].set_index(["dataset", "method"])
    classification = frames["classification"]
    lines = []
    for d in TARGETS:
        tag = " (UNDERPOWERED)" if classification[classification["dataset"] == d]["underpowered"].any() else ""
        def row(m):
            return seed_mean.loc[(d, m)]
        p1, a1, b5 = row("P1 Proposed head"), row("A1 Features-only head"), row("B5 Logits-only head")
        not_better = lambda r: "SUPPORTED (does not beat MSP)" if r["ci_high"] >= 0 else "NOT SUPPORTED (beats MSP)"
        lines.append(f"[{d}] H1 primary  P1 vs MSP  mean dAURC {p1['mean_delta_aurc_vs_msp']:+.4f} "
                     f"[{p1['ci_low']:+.4f}, {p1['ci_high']:+.4f}] -> {not_better(p1)}{tag}")
        lines.append(f"[{d}] H2          A1 vs MSP  mean dAURC {a1['mean_delta_aurc_vs_msp']:+.4f} "
                     f"[{a1['ci_low']:+.4f}, {a1['ci_high']:+.4f}] -> {not_better(a1)}{tag}")
        reg = frames["regime"][frames["regime"]["dataset"] == d]
        collapse = float(reg["top_class_share"].mean())
        if collapse >= COLLAPSE_THRESHOLD:
            v = "SUPPORTED" if b5["ci_high"] < 0 else ("CONTRADICTED" if b5["ci_low"] > 0 else "NOT SUPPORTED (inconclusive)")
            pred = "collapse -> B5 beats MSP"
        else:
            v = "SUPPORTED" if b5["ci_high"] >= 0 else "NOT SUPPORTED (B5 beats MSP without collapse)"
            pred = "no collapse -> B5 does not beat MSP"
        lines.append(f"[{d}] H3          collapse index {collapse:.3f} ({pred}); B5 vs MSP mean dAURC "
                     f"{b5['mean_delta_aurc_vs_msp']:+.4f} [{b5['ci_low']:+.4f}, {b5['ci_high']:+.4f}] -> {v}{tag}")
        cal = frames["calibration"][frames["calibration"]["dataset"] == d]
        worse = float(cal["nll_after"].mean()) >= float(cal["nll_before"].mean())
        lines.append(f"[{d}] H4          seed-mean NLL before {cal['nll_before'].mean():.4f}, after frozen temperature "
                     f"{cal['nll_after'].mean():.4f} -> {'SUPPORTED (no improvement)' if worse else 'NOT SUPPORTED (temperature helps)'}")
        al = frames["alarm"][frames["alarm"]["dataset"] == d]
        lines.append(f"[{d}] H5          S2 alarm rate >= 0.95 in every seed and batch size -> "
                     f"{'SUPPORTED' if (al['S2_alarm_rate'] >= 0.95).all() else 'NOT SUPPORTED'} (lowest {al['S2_alarm_rate'].min():.3f})")
        mech = frames["mechanism"].set_index("dataset").loc[d]
        v = "SUPPORTED" if mech["ci_low"] > 0 else ("CONTRADICTED" if mech["ci_high"] < 0 else "INCONCLUSIVE")
        lines.append(f"[{d}] H6          rho(B5,E) - rho(B5,-margin) {mech['seed_mean_gap']:+.4f} "
                     f"[{mech['ci_low']:+.4f}, {mech['ci_high']:+.4f}] -> {v}{tag}")
    return lines


def main():
    global OUT
    dry_run = len(sys.argv) == 3 and sys.argv[1] == "--dry-run-random-labels"
    if dry_run:
        OUT = Path(sys.argv[2])
        OUT.mkdir(parents=True, exist_ok=True)
        print("DRY RUN with random labels (real labels not read).")
    else:
        precheck()
        OUT.mkdir(parents=True, exist_ok=True)
        print("Joining the labels now.")
    thresholds = pd.read_csv(RESULTS_DIR / "source_alarm_thresholds.csv")
    alarm_thresholds = {(int(r.seed), int(r.batch_size)): {"S1_threshold": r.S1_threshold, "S2_threshold": r.S2_threshold}
                        for r in thresholds.itertuples()}

    collected = {}
    for dataset in DATASETS:
        if dry_run:
            paths = pd.read_csv(predictions_path(dataset, SEEDS[0]), keep_default_na=False)["path"]
            labels = pd.Series(np.random.default_rng(0).choice(CLASSES, len(paths)), index=paths)
        else:
            labels = labels_for(dataset)
        result = evaluate_dataset(dataset, load_tables(dataset, labels), alarm_thresholds)
        for key, rows in result.items():
            collected.setdefault(key, []).extend(rows)
    frames = {k: pd.DataFrame(v) for k, v in collected.items()}
    lines = verdicts(frames)
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 80)
    text = ["DINOv3 EVALUATION OF THE FROZEN SYSTEM (one-time, as pre-registered)", "=" * 90, "",
            "PRE-REGISTERED HYPOTHESES", *lines, ""]
    for name in ("classification", "confusion", "calibration", "regime", "seed_mean", "reliability", "mechanism",
                 "alarm", "decomposition", "magnitude"):
        text += [name.upper() + (" (exploratory)" if name in ("decomposition", "magnitude") else ""),
                 frames[name].round(4).to_string(index=False), ""]
    text.append("Nothing may be changed in the system, thresholds or analysis in response to these results.")
    (OUT / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    hashes = [f"{sha256(p)}  {p.name}" for p in sorted(OUT.glob("*")) if p.name != "evaluation_hashes.txt"]
    (OUT / "evaluation_hashes.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text[:4 + len(lines)]))


if __name__ == "__main__":
    main()
