"""
=====================================================================
 Brain Tumor Detection - Ensemble training and nested-CV evaluation
 ---------------------------------------------------------------
 An ensemble averages P(tumor) over several models. Members are trained
 on the same pool of images, but each with a different random (stratified)
 validation split and a different seed, so each member keeps almost all
 the training data and the members still differ.

 Evaluating an ensemble honestly needs NESTED cross-validation: the 5
 models of cross_validate.py have each seen the other folds' test images,
 so they cannot be averaged and scored on those folds. Instead:

   for each outer fold (the same folds as cross_validate.py):
       pool = all other folds
       train ENSEMBLE_SIZE members on the pool (val split from the pool)
       score the averaged prediction on the outer test fold

 Member 0 uses exactly the split and seed of cross_validate.py, so its
 scores equal the single-model CV and give a paired comparison on the
 same test images.

 Run:
   python ensemble.py
=====================================================================
"""

import os
import json
import tempfile

import numpy as np
from tensorflow import keras
from config import (SEED, VAL_FRACTION, CV_FOLDS, ENSEMBLE_SIZE,
                    ENSEMBLE_RESULTS_PATH, BACKBONE, IMG_SIZE)
from train import (prepare_dataset, training_datasets, make_dataset,
                   train_model, compute_metrics, member_split, make_folds,
                   per_source_metrics, display_path)

METRIC_NAMES = ["accuracy", "roc_auc", "sensitivity", "specificity"]


def train_ensemble(pool_paths, pool_labels, pool_sources, n_members, val_size,
                   model_paths, seed_offset=0, verbose=2):
    """Train n_members models on the pool; member i is saved to model_paths[i]."""
    for i in range(n_members):
        print(f"\n--- Ensemble member {i + 1}/{n_members} ---")
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + seed_offset + 1000 * i)
        # Member 0's split (random_state=SEED) equals the one in cross_validate.py.
        train_paths, val_paths, train_labels, val_labels = member_split(
            pool_paths, pool_labels, i, val_size, pool_sources)
        print(f"Images -> train: {len(train_paths)}, val: {len(val_paths)}")
        train_model(*training_datasets(train_paths, train_labels, val_paths, val_labels),
                    model_paths[i], verbose=verbose)


def predict_probs(model_path, paths, labels):
    model = keras.models.load_model(model_path)
    return model.predict(make_dataset(paths, labels, shuffle=False),
                         verbose=0).ravel()


def summarize(values_per_fold):
    values = np.array(values_per_fold, dtype="float64")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def main():
    keras.utils.set_random_seed(SEED)

    all_paths, all_labels, all_sources, data_info = prepare_dataset()

    val_size = VAL_FRACTION / (1.0 - 1.0 / CV_FOLDS)

    # Resume support: finished folds are checkpointed after each fold, so an
    # interrupted run (sleep, shutdown) continues where it stopped. A
    # checkpoint is only reused if it was made for exactly the same setup.
    checkpoint_path = ENSEMBLE_RESULTS_PATH + ".partial"
    setup = {"backbone": BACKBONE, "img_size": IMG_SIZE, "ensemble_size": ENSEMBLE_SIZE,
             "n_folds": CV_FOLDS, "images": [display_path(p) for p in all_paths],
             "labels": all_labels.tolist()}
    folds, oof = [], []          # oof: per-image out-of-fold predictions
    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["setup"] == setup:
            folds, oof = saved["folds"], saved["oof"]
            print(f"Resuming: {len(folds)} finished fold(s) loaded from {checkpoint_path}")
    except (OSError, ValueError, KeyError):
        pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        for k, (pool_idx, test_idx) in enumerate(
                make_folds(all_labels, all_sources, CV_FOLDS), 1):
            if k <= len(folds):
                continue                                  # done in an earlier run
            print(f"\n{'=' * 60}\n Outer fold {k}/{CV_FOLDS}\n{'=' * 60}")
            test_paths, test_labels = all_paths[test_idx], all_labels[test_idx]
            model_paths = [os.path.join(tmp_dir, f"fold{k}_member{i}.keras")
                           for i in range(ENSEMBLE_SIZE)]
            # seed_offset=k: member 0 then uses SEED + k, as in cross_validate.py
            train_ensemble(all_paths[pool_idx], all_labels[pool_idx],
                           all_sources[pool_idx], ENSEMBLE_SIZE, val_size,
                           model_paths, seed_offset=k)

            member_probs = [predict_probs(p, test_paths, test_labels)
                            for p in model_paths]
            members = [compute_metrics(test_labels, pr, 0.5) for pr in member_probs]
            ensemble = compute_metrics(test_labels, np.mean(member_probs, axis=0), 0.5)
            folds.append({"fold": k, "n_test": int(len(test_idx)),
                          "ensemble": ensemble, "members": members,
                          "ensemble_per_source": per_source_metrics(
                              test_labels, np.mean(member_probs, axis=0),
                              all_sources[test_idx], 0.5)})
            for j, path in enumerate(test_paths):
                member_p = [float(pr[j]) for pr in member_probs]
                oof.append({"path": display_path(path),
                            "source": str(all_sources[test_idx][j]),
                            "label": int(test_labels[j]), "fold": k,
                            "p_tumor": float(np.mean(member_p)),
                            "p_tumor_members": member_p})
            print(f"Fold {k}: ensemble " + ", ".join(
                f"{m}: {ensemble[m]:.4f}" for m in METRIC_NAMES))
            print(f"Fold {k}: member 0 " + ", ".join(
                f"{m}: {members[0][m]:.4f}" for m in METRIC_NAMES))
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"setup": setup, "folds": folds, "oof": oof}, f)

    summary = {
        "ensemble": {m: summarize([f["ensemble"][m] for f in folds]) for m in METRIC_NAMES},
        "single_model": {m: summarize([f["members"][0][m] for f in folds]) for m in METRIC_NAMES},
        "mean_of_members": {m: summarize([np.mean([x[m] for x in f["members"]]) for f in folds])
                            for m in METRIC_NAMES},
        "paired_gain_vs_single": {m: summarize([f["ensemble"][m] - f["members"][0][m] for f in folds])
                                  for m in METRIC_NAMES},
    }

    print(f"\n{'=' * 60}\n Nested CV: ensemble of {ENSEMBLE_SIZE} vs single model "
          f"({len(all_paths)} distinct images)\n{'=' * 60}")
    print(f"{'metric':>12} {'single (member 0)':>20} {'ensemble':>18} {'paired gain':>16}")
    for m in METRIC_NAMES:
        s, e, g = (summary[key][m] for key in ("single_model", "ensemble", "paired_gain_vs_single"))
        print(f"{m:>12} {s['mean']:>10.4f} +/- {s['std']:.4f} {e['mean']:>8.4f} +/- {e['std']:.4f} "
              f"{g['mean']:>+8.4f} +/- {g['std']:.4f}")

    results = {"n_folds": CV_FOLDS, "ensemble_size": ENSEMBLE_SIZE,
               "backbone": BACKBONE, "img_size": IMG_SIZE,
               **data_info,
               "n_images": int(len(all_paths)),
               "std": "sample standard deviation over folds (ddof=1)",
               "summary": summary, "folds": folds,
               "out_of_fold_predictions": oof}
    with open(ENSEMBLE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.remove(checkpoint_path)
    print("\nNested-CV results saved to:", ENSEMBLE_RESULTS_PATH)


if __name__ == "__main__":
    main()
