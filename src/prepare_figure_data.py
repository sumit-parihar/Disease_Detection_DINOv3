"""
prepare_figure_data.py  -  Tables behind the figures of the combined study (MobileNetV3-Small and DINOv3,
PP2020 and PlantDoc). Reads only saved, already-evaluated outputs of the three projects; nothing is
re-fitted. Accuracy, collapse index, NLL and risk-coverage curves are recomputed from the saved
predictions with the ORIGINAL metrics.py, so both backbones are summarised by the same code.

OUTPUT figure_data/
  G1_cross_study_delta_aurc.csv     backbone, dataset, method, mean dAURC vs MSP and 95% interval
  G2_accuracy_collapse.csv          backbone, dataset, seed, accuracy, macro-F1, errors, collapse index
  G3_synthetic_view_failures.csv    backbone, seed, reliability split, view type, failing views / views
  G4_calibration.csv                backbone, dataset, seed, temperature, NLL / ECE / Brier before and after
  G5_batch_alarm.csv                backbone, dataset, seed, batch size, S2 alarm rate
  G6_risk_coverage_dinov3.csv       DINOv3 risk-coverage curves (dataset, seed, method, coverage, risk)
  G7_auroc_decomposition.csv        backbone, dataset, seed, score, overall / between / within AUROC-error
"""

import json

import numpy as np
import pandas as pd

from common import EXTENDED_ROOT, FROZEN_ROOT, RESULTS_DIR, ROOT, SEEDS, use_frozen_code

use_frozen_code()
from metrics import calibration_summary, classification_summary, risk_coverage_curve   # noqa: E402

OUT = ROOT / "figure_data"
CLASSES = ["healthy", "scab", "rust"]
LOGITS = ["logit_healthy", "logit_scab", "logit_rust"]
CURVE_METHODS = {"B1 MSP": "msp_score", "B4 Temp-scaled MSP": "temp_scaled_msp_score",
                 "B5 Logits-only head": "head_logits_only", "A1 Features-only head": "head_features_only",
                 "P1 Proposed head": "head_proposed"}
MOBILENET_SETTINGS = json.loads((FROZEN_ROOT / "results" / "frozen_calibration_settings.json").read_text(encoding="utf-8"))
DINO_SETTINGS = json.loads((RESULTS_DIR / "frozen_calibration_settings.json").read_text(encoding="utf-8"))


def labels(dataset):
    if dataset == "PP2020":
        m = pd.read_csv(FROZEN_ROOT / "manifests" / "manifest.csv", keep_default_na=False)
        return m[m["dataset"] == "target"].set_index("path")["label"]
    if dataset == "PlantDoc":
        return pd.read_csv(EXTENDED_ROOT / "manifests" / "plantdoc_eligible.csv", keep_default_na=False).set_index("staged_path")["label"]
    split = pd.read_csv(FROZEN_ROOT / "splits" / "split_assignments.csv", keep_default_na=False)
    return split.set_index("path")["label"]


def predictions(backbone, dataset, seed):
    if backbone == "DINOv3":
        path = RESULTS_DIR / "inference" / f"{dataset.lower().replace('-', '_')}_predictions_seed{seed}.csv"
    elif dataset == "PP2020":
        path = FROZEN_ROOT / "results" / f"target_predictions_seed{seed}.csv"
    elif dataset == "PlantDoc":
        path = EXTENDED_ROOT / "results" / "plantdoc_inference" / f"plantdoc_predictions_seed{seed}.csv"
    else:
        path = FROZEN_ROOT / "results" / f"s_test_outputs_seed{seed}.csv"
    t = pd.read_csv(path, keep_default_na=False)
    lab = labels(dataset)
    t["true_index"] = [CLASSES.index(lab[p]) for p in t["path"]]
    t["error"] = (t["predicted_index"] != t["true_index"]).astype(int)
    return t


def main():
    OUT.mkdir(exist_ok=True)
    combos = [("MobileNetV3-Small", d) for d in ("S-test", "PP2020", "PlantDoc")] + \
             [("DINOv3 ViT-S/16", d) for d in ("S-test", "PP2020", "PlantDoc")]

    # G2 and G4
    g2, g4 = [], []
    for backbone, dataset in combos:
        key = "DINOv3" if backbone.startswith("DINOv3") else "MobileNet"
        settings = DINO_SETTINGS if key == "DINOv3" else MOBILENET_SETTINGS
        for seed in SEEDS:
            t = predictions(key, dataset, seed)
            y, p = t["true_index"].to_numpy(), t["predicted_index"].to_numpy()
            s = classification_summary(y, p, CLASSES)
            shares = np.bincount(p, minlength=3) / len(p)
            g2.append({"backbone": backbone, "dataset": dataset, "seed": seed, "images": len(t), "errors": int(t["error"].sum()),
                       "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "collapse_index": shares.max(),
                       "most_predicted_class": CLASSES[int(shares.argmax())]})
            logits = t[LOGITS].to_numpy(dtype=np.float64)
            temperature = settings[str(seed)]["temperature"]
            before, after = calibration_summary(logits, y), calibration_summary(logits, y, temperature)
            g4.append({"backbone": backbone, "dataset": dataset, "seed": seed, "temperature": temperature,
                       **{f"{k}_before": v for k, v in before.items()}, **{f"{k}_after": v for k, v in after.items()}})
    pd.DataFrame(g2).to_csv(OUT / "G2_accuracy_collapse.csv", index=False, lineterminator="\n")
    pd.DataFrame(g4).to_csv(OUT / "G4_calibration.csv", index=False, lineterminator="\n")

    # G1
    g1 = []
    orig = pd.read_csv(FROZEN_ROOT / "results" / "target_reliability_seed_mean.csv")
    for r in orig.itertuples():
        g1.append({"backbone": "MobileNetV3-Small", "dataset": "PP2020", "method": r.method,
                   "mean_delta_aurc": r.mean_delta_aurc, "ci_low": r.mean_delta_low, "ci_high": r.mean_delta_high,
                   "mean_auroc_error": r.mean_auroc_error})
    ext = pd.read_csv(EXTENDED_ROOT / "results" / "plantdoc_evaluation" / "reliability_seed_mean.csv")
    for r in ext.itertuples():
        g1.append({"backbone": "MobileNetV3-Small", "dataset": "PlantDoc", "method": r.method.replace(" (pre-registered)", ""),
                   "mean_delta_aurc": r.mean_delta_aurc_vs_msp, "ci_low": r.ci_low, "ci_high": r.ci_high,
                   "mean_auroc_error": r.mean_auroc_error})
    dino = pd.read_csv(RESULTS_DIR / "evaluation" / "seed_mean.csv")
    for r in dino[dino["dataset"] != "s_test"].itertuples():
        g1.append({"backbone": "DINOv3 ViT-S/16", "dataset": "PP2020" if r.dataset == "pp2020" else "PlantDoc",
                   "method": r.method, "mean_delta_aurc": r.mean_delta_aurc_vs_msp, "ci_low": r.ci_low,
                   "ci_high": r.ci_high, "mean_auroc_error": r.mean_auroc_error})
    pd.DataFrame(g1).to_csv(OUT / "G1_cross_study_delta_aurc.csv", index=False, lineterminator="\n")

    # G3
    g3 = []
    for backbone, pattern in (("MobileNetV3-Small", FROZEN_ROOT / "results" / "reliability_v2_predictions_seed{}.csv"),
                              ("DINOv3 ViT-S/16", RESULTS_DIR / "reliability_predictions_seed{}.csv")):
        for seed in SEEDS:
            t = pd.read_csv(str(pattern).format(seed), keep_default_na=False)
            grouped = t.groupby(["reliability_split", "view_type"])["failure_label"].agg(["sum", "count"]).reset_index()
            for r in grouped.itertuples():
                g3.append({"backbone": backbone, "seed": seed, "reliability_split": r.reliability_split,
                           "view_type": r.view_type, "failing_views": int(r.sum), "views": int(r.count)})
    pd.DataFrame(g3).to_csv(OUT / "G3_synthetic_view_failures.csv", index=False, lineterminator="\n")

    # G5
    g5 = []
    part3 = pd.read_csv(EXTENDED_ROOT / "results" / "exploratory_pp2020" / "part3_batch_alarm.csv")
    for r in part3.to_dict("records"):
        g5.append({"backbone": "MobileNetV3-Small", "dataset": "PP2020", "seed": r["seed"], "batch_size": r["batch_size"],
                   "S2_alarm_rate": r["S2 alarm rate, PP2020 batches"], "status": "exploratory"})
        g5.append({"backbone": "MobileNetV3-Small", "dataset": "S-test", "seed": r["seed"], "batch_size": r["batch_size"],
                   "S2_alarm_rate": r["S2 false alarms, S-test"], "status": "exploratory (false alarms)"})
    for r in pd.read_csv(EXTENDED_ROOT / "results" / "plantdoc_evaluation" / "batch_alarm.csv").to_dict("records"):
        g5.append({"backbone": "MobileNetV3-Small", "dataset": "PlantDoc", "seed": r["seed"], "batch_size": r["batch_size"],
                   "S2_alarm_rate": r["S2 alarm rate"], "status": "pre-registered"})
    names = {"s_test": "S-test", "pp2020": "PP2020", "plantdoc": "PlantDoc"}
    for r in pd.read_csv(RESULTS_DIR / "evaluation" / "alarm.csv").to_dict("records"):
        g5.append({"backbone": "DINOv3 ViT-S/16", "dataset": names[r["dataset"]], "seed": r["seed"],
                   "batch_size": r["batch_size"], "S2_alarm_rate": r["S2_alarm_rate"],
                   "status": "false alarms" if r["dataset"] == "s_test" else "pre-registered"})
    pd.DataFrame(g5).to_csv(OUT / "G5_batch_alarm.csv", index=False, lineterminator="\n")

    # G6
    curves = []
    for dataset in ("PP2020", "PlantDoc"):
        for seed in SEEDS:
            t = predictions("DINOv3", dataset, seed)
            for method, col in CURVE_METHODS.items():
                coverage, risk = risk_coverage_curve(t["error"], t[col])
                curves.append(pd.DataFrame({"dataset": dataset, "seed": seed, "method": method,
                                            "coverage": coverage, "selective_risk": risk}))
    pd.concat(curves).to_csv(OUT / "G6_risk_coverage_dinov3.csv", index=False, lineterminator="\n")

    # G7
    mob = pd.read_csv(EXTENDED_ROOT / "results" / "exploratory_regimes" / "auroc_decomposition.csv")
    mob = pd.DataFrame({"backbone": "MobileNetV3-Small", "dataset": mob["dataset"], "seed": mob["seed"], "score": mob["score"],
                        "auroc_overall": mob["AUROC overall"], "auroc_between_classes": mob["AUROC between classes only"],
                        "auroc_within_weighted": mob["AUROC within classes (weighted)"]})
    dec = pd.read_csv(RESULTS_DIR / "evaluation" / "decomposition.csv")
    dec = dec[dec["dataset"] != "s_test"]
    short = {"B1 MSP": "MSP", "B5 Logits-only head": "B5", "E Energy": "Energy", "P1 Proposed head": "P1"}
    dec = pd.DataFrame({"backbone": "DINOv3 ViT-S/16", "dataset": dec["dataset"].map(names), "seed": dec["seed"],
                        "score": dec["score"].map(short), "auroc_overall": dec["auroc_overall"],
                        "auroc_between_classes": dec["auroc_between_classes"], "auroc_within_weighted": dec["auroc_within_weighted"]})
    pd.concat([mob, dec]).to_csv(OUT / "G7_auroc_decomposition.csv", index=False, lineterminator="\n")

    summary = pd.DataFrame(g2).groupby(["backbone", "dataset"])[["accuracy", "collapse_index", "errors"]].mean().round(3)
    print(summary)
    print(pd.DataFrame(g4).groupby(["backbone", "dataset"])[["temperature", "nll_before", "nll_after"]].mean().round(3))


if __name__ == "__main__":
    main()
