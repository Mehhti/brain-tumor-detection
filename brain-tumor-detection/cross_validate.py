"""
=====================================================================
 Brain Tumor Detection - Stratified k-Fold Cross-Validation
 ---------------------------------------------------------------
 After duplicate removal Br35H has only ~620 distinct scans, so a
 single 10% test set (~60 images) gives a noisy estimate. This script
 evaluates the SAME pipeline as train.py on every image instead:

   for each of CV_FOLDS stratified folds:
       test  = this fold (never seen during training)
       val   = stratified VAL_FRACTION (of all images) from the rest,
               used only for early stopping / checkpointing and for
               tuning the decision threshold
       train = everything else
       -> train both stages from scratch, reload the saved best model,
          score it on the test fold at threshold 0.5 and at the
          validation-tuned threshold

 Reports mean +/- std (sample std over folds) of accuracy, ROC AUC,
 sensitivity and specificity for both thresholds, and writes all numbers
 to CV_RESULTS_PATH.

 Because every image is predicted exactly once by a model that never
 saw it, the out-of-fold predictions are also used to flag SUSPECTED
 LABEL ISSUES: images the model confidently assigns to the other class.
 These are candidates for manual review, not proven errors.

 The model used by the web app is still produced by train.py.

 Run:
   python cross_validate.py
=====================================================================
"""

import os
import json
import tempfile

import numpy as np
from tensorflow import keras
from config import SEED, CLASS_NAMES, VAL_FRACTION, CV_FOLDS, CV_RESULTS_PATH
from train import (prepare_dataset, training_datasets, train_model,
                   evaluate_saved_model, make_folds, member_split,
                   per_source_metrics, display_path)

METRIC_NAMES = ["accuracy", "roc_auc", "sensitivity", "specificity"]

# An out-of-fold prediction this far on the wrong side counts as a
# suspected label issue (e.g. labelled "no" but P(tumor) >= 0.9).
LABEL_ISSUE_CONFIDENCE = 0.9


def summarize(folds, key):
    """Mean and sample std over folds of metrics stored under folds[i][key]."""
    summary = {}
    for m in METRIC_NAMES:
        values = np.array([f[key][m] for f in folds], dtype="float64")
        summary[m] = {"mean": float(values.mean()),
                      "std": float(values.std(ddof=1))}
    return summary


def main():
    keras.utils.set_random_seed(SEED)

    # -----------------------------------------------------------
    # 1) Same data preparation as train.py (duplicates removed first)
    # -----------------------------------------------------------
    all_paths, all_labels, all_sources, data_info = prepare_dataset()

    # -----------------------------------------------------------
    # 2) Stratified k-fold: every image is in the test fold exactly once.
    #    Folds are made within each source, so adding a source does not
    #    change the folds of the others (paired comparisons stay valid).
    # -----------------------------------------------------------
    # VAL_FRACTION is relative to all images; convert to the train part.
    val_size = VAL_FRACTION / (1.0 - 1.0 / CV_FOLDS)

    folds = []
    oof_prob = np.full(len(all_paths), np.nan)          # out-of-fold P(tumor)
    oof_fold = np.zeros(len(all_paths), dtype="int32")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for k, (rest_idx, test_idx) in enumerate(
                make_folds(all_labels, all_sources, CV_FOLDS), 1):
            print(f"\n{'=' * 60}\n Fold {k}/{CV_FOLDS}\n{'=' * 60}")
            keras.backend.clear_session()
            keras.utils.set_random_seed(SEED + k)

            # Same split as ensemble member 0 (stratified by source x label).
            train_paths, val_paths, train_labels, val_labels = member_split(
                all_paths[rest_idx], all_labels[rest_idx], 0, val_size,
                all_sources[rest_idx])
            test_paths, test_labels = all_paths[test_idx], all_labels[test_idx]

            # Sanity check: the three sets must be disjoint.
            assert not set(train_paths) & set(val_paths)
            assert not set(train_paths) & set(test_paths)
            assert not set(val_paths) & set(test_paths)
            print(f"Images -> train: {len(train_paths)}, val: {len(val_paths)}, "
                  f"test: {len(test_paths)}")

            model_path = os.path.join(tmp_dir, f"fold{k}.keras")
            train_model(*training_datasets(train_paths, train_labels,
                                           val_paths, val_labels),
                        model_path, verbose=2)

            test_prob, tuned, at_default, at_tuned = evaluate_saved_model(
                model_path, val_paths, val_labels, test_paths, test_labels)
            oof_prob[test_idx] = test_prob
            oof_fold[test_idx] = k

            folds.append({"fold": k, "n_train": int(len(train_paths)),
                          "n_val": int(len(val_paths)),
                          "n_test": int(len(test_paths)),
                          "tuned_threshold": tuned,
                          "threshold_0.5": at_default,
                          "val_tuned": at_tuned,
                          "per_source": per_source_metrics(
                              test_labels, test_prob, all_sources[test_idx], 0.5)})
            for name, m in (("0.5", at_default), (f"{tuned:.3f}", at_tuned)):
                print(f"Fold {k} test @ {name:>5} -> " + ", ".join(
                    f"{x}: {m[x]:.4f}" for x in METRIC_NAMES))

    # -----------------------------------------------------------
    # 3) Summary: mean +/- std over folds, for both thresholds
    # -----------------------------------------------------------
    summary = {"threshold_0.5": summarize(folds, "threshold_0.5"),
               "val_tuned": summarize(folds, "val_tuned")}
    thresholds = np.array([f["tuned_threshold"] for f in folds])

    print(f"\n{'=' * 60}\n {CV_FOLDS}-fold cross-validation results "
          f"({len(all_paths)} distinct images)\n{'=' * 60}")
    for key, title in (("threshold_0.5", "threshold 0.5"),
                       ("val_tuned", "threshold tuned on each fold's validation set")):
        print(f"\n[{title}]")
        print(f"{'fold':>4} {'thr':>6} {'n_test':>6} " +
              " ".join(f"{m:>11}" for m in METRIC_NAMES))
        for f in folds:
            print(f"{f['fold']:>4} {f[key]['threshold']:>6.3f} {f['n_test']:>6} " +
                  " ".join(f"{f[key][m]:>11.4f}" for m in METRIC_NAMES))
        for m in METRIC_NAMES:
            print(f"{m:>12}: {summary[key][m]['mean']:.4f} +/- {summary[key][m]['std']:.4f}")
    print(f"\nTuned thresholds: mean {thresholds.mean():.3f}, "
          f"range {thresholds.min():.3f} - {thresholds.max():.3f}")

    # Per source (threshold 0.5): mean +/- std over folds
    summary["per_source"] = {}
    for name in sorted(set.intersection(*(set(f["per_source"]) for f in folds))):
        summary["per_source"][name] = summarize(
            [{"m": f["per_source"][name]} for f in folds], "m")
        print(f"\n[{name}] " + ", ".join(
            f"{m}: {summary['per_source'][name][m]['mean']:.4f} +/- "
            f"{summary['per_source'][name][m]['std']:.4f}" for m in METRIC_NAMES))

    # -----------------------------------------------------------
    # 4) Suspected label issues (confident out-of-fold disagreements)
    # -----------------------------------------------------------
    disagreement = np.where(all_labels == 1, 1.0 - oof_prob, oof_prob)
    suspects = np.flatnonzero(disagreement >= LABEL_ISSUE_CONFIDENCE)
    suspects = suspects[np.argsort(-disagreement[suspects])]
    label_issues = [{"path": display_path(all_paths[i]),
                     "label": CLASS_NAMES[all_labels[i]],
                     "p_tumor": float(oof_prob[i]),
                     "fold": int(oof_fold[i])} for i in suspects]

    print(f"\nSuspected label issues (out-of-fold P(tumor) on the wrong side "
          f"of {LABEL_ISSUE_CONFIDENCE:.0%}): {len(label_issues)}")
    for item in label_issues[:15]:
        print(f"  {item['path']:<45} label={item['label']:<3} "
              f"P(tumor)={item['p_tumor']:.3f}")

    results = {
        "n_folds": CV_FOLDS,
        **data_info,
        "n_images": int(len(all_paths)),
        "std": "sample standard deviation over folds (ddof=1)",
        "summary": summary,
        "tuned_thresholds": thresholds.tolist(),
        "folds": folds,
        "suspected_label_issues": label_issues,
        "out_of_fold_predictions": [
            {"path": display_path(p), "source": str(so), "label": int(y),
             "p_tumor": float(pr), "fold": int(fo)}
            for p, so, y, pr, fo in zip(all_paths, all_sources, all_labels,
                                        oof_prob, oof_fold)],
    }
    with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nCross-validation results saved to:", CV_RESULTS_PATH)


if __name__ == "__main__":
    main()
