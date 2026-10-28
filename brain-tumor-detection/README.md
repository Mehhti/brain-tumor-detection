# 🧠 Brain Tumor Detection with Deep Learning

Detect brain tumors from MRI images using two-stage **transfer learning**
(an ensemble of three EfficientNetV2-B0 models) trained on two public
datasets, served through a simple **Streamlit** web app.

The project covers the full workflow: data loading, model training with a
proper train/validation/**independent-test** split, honest evaluation, and an
interactive app that also validates whether the uploaded image is actually a
brain-MRI scan.

> ⚠️ Educational / research project. Not a medical device and not validated
> on clinical data; see [Limitations](#limitations).

## 🔎 Key findings

- **Br35H is mostly duplicates.** Of its 3000 images only **419 are distinct
  scans**; the rest are the same scans re-saved with other brightness,
  contrast, crop, or mirrored, flipped or rotated (one scan appears 61 times).
  Without removing them, copies of training scans end up in the test set and
  accuracy is inflated. Published ~98–99% accuracies on this dataset should be
  read with that in mind.
- **Honest accuracy is lower, and defensible.** On the 416 distinct Br35H
  scans the final model reaches **85.3%** accuracy (AUC 0.931) in nested
  cross-validation; on a second dataset (Sartaj, 1878 distinct scans) **96.1%**
  (AUC 0.974).
- **Adding a second dataset can make a model worse.** With plain class weights
  the model learned "Br35H-looking = healthy" and missed a third of Br35H
  tumors; **balanced source × label weighting** fixed this, and a model trained
  on Br35H alone generalised poorly to the other dataset (AUC 0.76 vs 0.97).
- Every model choice (backbone, ensemble, threshold, data) was made with
  **paired cross-validation**, including the experiments that did not help;
  they are reported below.

---

## ✨ Features

- **Transfer learning** with EfficientNetV2-B0 at 224 px (pretrained on
  ImageNet); MobileNetV2 is also supported (`BACKBONE` in `config.py`). The
  backbone was chosen by cross-validation (see Results).
- **Two-stage training:** train the classifier head, then fine-tune the top layers.
- **Two data sources** (Br35H and Sartaj Bhuvaji's dataset), deduplicated
  across sources and trained with **balanced source × label weighting**, so
  the model cannot use "which dataset does this look like" as a shortcut.
- **Ensemble of 3 models** (`ENSEMBLE_SIZE`), each with its own validation
  split and seed, averaged in the app. This removes the run-to-run luck of a
  single training run.
- **Duplicate removal:** duplicate and near-duplicate images (including
  mirrored, upside-down and 180°-rotated copies) are detected with a
  perceptual hash and removed **before** splitting, so copies of one scan
  can't leak from training into the test set.
- **Label review** (`label_review.json`): images whose label is doubtful can be
  excluded or relabelled; the decision applies to all copies of that scan.
- **Leak-free 3-way split** (stratified 80 / 10 / 10 on the file list) so the
  reported score is measured on data the model never saw.
- **Honest evaluation:** the saved best model is reloaded and scored on the test
  set (accuracy, ROC AUC, sensitivity, specificity, ROC curve).
- **Class weighting** for robustness against class imbalance.
- **Cross-validation** (stratified 5-fold) for a reliable performance estimate,
  plus a list of **suspected label issues** found from out-of-fold predictions.
- **Grad-CAM** in the web app: for a "tumor" result, a heatmap shows which
  region of the scan drove the prediction.
- **Input validation** in the web app: quick checks reject color photos,
  documents and unreadable files; a **feature-space check** then flags images
  that are unlike every training scan (e.g. CT, X-ray, photos, unseen MRI
  views); low-confidence predictions are flagged too. Uploads are decoded and
  resized exactly like the training images.
- **Expert review sheet** (`make_review_sheet.py`): one HTML page with every
  image whose label is in doubt, for a qualified reader to decide.
- **Automated tests** (`pytest`) for duplicate detection, label review, data
  sources and weighting, input checks, the out-of-distribution score,
  threshold selection, preprocessing parity and Grad-CAM.
- Clean, configurable code (all settings live in `config.py`).

---

## 📁 Project structure

```
brain-tumor-detection/
├── app.py             # Streamlit web app
├── train.py           # Model training script
├── cross_validate.py  # Stratified k-fold cross-validation of the pipeline
├── ensemble.py        # Nested cross-validation of the ensemble
├── backbones.py       # Supported pretrained backbones
├── inference.py       # App inference: preprocessing, prediction, Grad-CAM
├── image_check.py     # Input-image suitability check
├── dedup.py           # Duplicate / near-duplicate detection (perceptual hash)
├── make_review_sheet.py  # HTML sheet of doubtful labels for an expert
├── config.py          # Central configuration (paths & hyperparameters)
├── label_review.json  # Manual label decisions (exclude / relabel)
├── tests/             # pytest test suite (no dataset or model needed)
├── .github/workflows/tests.yml  # runs the tests on GitHub Actions
├── pytest.ini
├── requirements.txt   # Python dependencies (pinned)
├── requirements-dev.txt  # + pytest
├── README.md
├── LICENSE
└── .gitignore
```

> Note: the datasets and the trained models are **not** included in the repo
> (see `.gitignore`). Download the datasets and train locally (`train.py`), or
> place trained model files (`best_model_br35h*.keras`, `ood_reference.npz`,
> `test_metrics.json`) next to `app.py`, e.g. from the repository's Releases
> page if they are published there. The result files (`cv_results.json`,
> `ensemble_cv_results.json`, `duplicates_report.json`, `test_metrics.json`)
> are included, so every number in this README can be checked.

---

## 📊 Data

Two public datasets are used (`USE_SOURCES` in `config.py`):

| Source | Download | Files | Distinct scans used |
|---|---|---|---|
| **Br35H :: Brain Tumor Detection 2020** | <https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection> | 3000 | 416 (163 *no*, 253 *yes*) |
| **Brain Tumor Classification (MRI)**, Sartaj Bhuvaji | <https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet> (also on Kaggle) | 3264 | 1878 (172 *no*, 1706 *yes*) |

Point `DATA_DIR` at the Br35H folder that directly contains `yes/` and `no/`
(a `pred/` folder is ignored), and `SARTAJ_DIR` at the Sartaj folder that
contains `Training/` and `Testing/`. In Sartaj, `no_tumor` is the healthy class
and the three tumor types (glioma, meningioma, pituitary) form the tumor class.
The Sartaj repository has no license file, so its terms of use are not stated;
the data is not part of this repository. To use Br35H alone, set
`USE_SOURCES = ["br35h"]`.

### ⚠️ Duplicates in Br35H

Br35H contains many copies of the same scans, re-saved with different
brightness, contrast, crop or zoom, and often mirrored left-right, flipped
upside-down or rotated by 180°. One healthy scan appears 61 times. With the
default settings (`config.py`: pHash distance ≤ 6, flipped / rotated copies
included), only **419 of the 3000 images are distinct scans** (166 *no* /
253 *yes*).

`train.py` removes these copies before splitting and writes every removed
file, grouped with the copy that was kept, to `duplicates_report.json`.
Without this step the same scan lands in both train and test, and the test
score mostly measures memorization. Expect lower, but honest, numbers.
The threshold was chosen by inspecting matched pairs: up to 6 they are almost
always the same scan, while from 8 on different scans (even across classes)
start to match. An earlier version only matched left-right mirrors; it missed
230 copy pairs (upside-down / rotated), which still inflated cross-validated
accuracy by about 4–5 points. Set `DEDUPLICATE = False` to reproduce the
original behaviour.

### 🏷️ Label review

`label_review.json` lists images whose label was reviewed, with an action
(`exclude`, `relabel` with a new `label`, or `keep`) and a reason. It is
applied before splitting, and a decision applies to the image's whole
duplicate group. Currently it excludes one scan stored under four file names
(`br35h/no/no1102.jpg`, `no384`, `no1323`, `no1281`): it is labelled *no* but
shows two round, strongly enhancing focal lesions. Paths are written as
`<source>/<path inside the source>`. The candidates came from confident
disagreements of two independent models with the label. These decisions are
provisional, not diagnoses; please have them confirmed or changed by a
qualified reader.

`python make_review_sheet.py` writes `review/label_review_sheet.html`: every
image in doubt (current decisions, confident model disagreements from
cross-validation, and scans stored under different labels) with thumbnails and
a field for the expert's decision. It embeds dataset images, so `review/` is
not tracked by git.

### ➕ Second dataset and source bias

Adding Sartaj's dataset needed care:

- **Copies across datasets.** Sartaj alone reduces from 3264 files to 2074
  distinct scans. Duplicate detection runs over both sources together: 195
  duplicate groups span both datasets (Sartaj's healthy class partly comes from
  Br35H), so only 172 of Sartaj's 283 distinct healthy scans are new.
- **Coincidental hash matches.** With ~6000 images in 4 orientations, some
  *different* scans fall within the hash distance by chance. When such a match
  joins images with different labels it would look like "the same scan,
  labelled both ways", and the first version reported 8 such conflicts. A
  visual check showed all 8 were different scans (e.g. a healthy scan next to
  a glioma), with pixel correlations of 0.60–0.91 versus about 1.0 for genuine
  copies. Cross-label matches are therefore double-checked on the pixels
  (`DUPLICATE_CONFLICT_MIN_CORR = 0.95`); no genuine conflict remains.
- **Source bias.** The combined data is very unbalanced by source: Br35H gives
  about half of all healthy scans but only 13% of the tumors, and Sartaj's
  tumors are often side or front views. Trained with plain class weights, the
  model learned "Br35H-looking = healthy": on the same Br35H images its
  errors rose from 61 to 97 of 413 (a third of Br35H tumors missed; McNemar
  p = 0.0003), while the overall accuracy looked excellent (92.5%) because
  Sartaj's many tumors are easy.
- **Fix: balanced weighting.** Every source × label group (Br35H healthy,
  Br35H tumor, Sartaj healthy, Sartaj tumor) gets the same total weight in
  training and validation. Then the combined model ties the Br35H-only model on
  Br35H (60 vs 61 errors, AUC 0.925 vs 0.918) and is far better on Sartaj,
  where a Br35H-only model reaches only AUC 0.760 and flags 3 of 4 healthy
  Sartaj scans as tumor (combined model: AUC 0.964).
- Cross-validation folds are made **per source**, so a source's folds do not
  change when another source is added, and results are reported per source.

## 🚀 Getting started

### 1. Clone and install

Requires **Python 3.10 – 3.13**. Dependency versions are pinned in
`requirements.txt` for reproducibility.

```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

### 2. Configure the data path

By default the data is read from `data/` (Br35H) and `data_sartaj/` (Sartaj)
next to the scripts. Edit `config.py` or set environment variables to use other
locations:

```bash
export DATA_DIR="/path/to/br35h"
export SARTAJ_DIR="/path/to/Brain-Tumor-Classification-DataSet"
```

On Windows (PowerShell):

```powershell
$env:DATA_DIR = "C:\path\to\br35h"
$env:SARTAJ_DIR = "C:\path\to\Brain-Tumor-Classification-DataSet"
```

### 3. Train the model

```bash
python train.py
```

This first removes duplicate images (see above, report in
`duplicates_report.json`), then trains `ENSEMBLE_SIZE` (= 3) models. Each one
keeps its best epoch (by validation AUC, across both stages):
`best_model_br35h.keras`, `best_model_br35h_2.keras`, `best_model_br35h_3.keras`.
On CPU this takes roughly 10–15 minutes. It then reloads the saved models, averages
their predictions and, on the **independent test set**:

- prints the confusion matrix, classification report and ROC AUC,
- writes the metrics to `test_metrics.json` (shown in the web app's sidebar),
- saves the ROC curve to `roc_curve.png`,

and saves the training curves to `training_history.png`. All outputs are
written next to the scripts.

### 4. (Optional) Cross-validate

```bash
python cross_validate.py
```

With only ~420 distinct scans, the single test set holds ~40 images, so its
score is noisy. This runs the **same pipeline** in stratified 5-fold
cross-validation (`CV_FOLDS` in `config.py`): each fold is the test set once,
a separate stratified validation split is taken from the remaining folds for
early stopping and checkpointing, and the model is trained from scratch per
fold. It prints per-fold results and mean ± std of accuracy, ROC AUC,
sensitivity and specificity, and saves them to `cv_results.json`. It takes
about five times as long as `train.py` and does not change the app's model.

Because every image is predicted once by a model that never saw it,
`cv_results.json` also lists **suspected label issues**: images the model
confidently assigns to the other class (P ≥ 0.9 against the label). These are
candidates for expert review, not proven errors; the list mixes possible
mislabels with genuine model mistakes.

To evaluate the **ensemble** honestly, use nested cross-validation:

```bash
python ensemble.py
```

For each outer fold it trains `ENSEMBLE_SIZE` models on the other folds and
scores their averaged prediction on the outer fold, and also reports each
member on its own (results in `ensemble_cv_results.json`). This trains
`CV_FOLDS × ENSEMBLE_SIZE` = 15 models (a few hours with EfficientNet on CPU).
Finished folds are checkpointed (`ensemble_cv_results.json.partial`), so if
the run is interrupted (sleep, shutdown), starting it again continues from the
last finished fold as long as the data and settings are unchanged.

### 5. Run the web app

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal (usually
`http://localhost:8501`) and upload an MRI image. The app loads all ensemble
members listed in `test_metrics.json` and averages their predictions. For a
"tumor" result it also shows a **Grad-CAM** heatmap (averaged over the
members). It is coarse (the backbone's last feature map is 7×7 at 224 px), so
it marks the rough area that drove the prediction rather than the exact tumor
outline. A higher-resolution layer (14×14) and Grad-CAM++ were tried: the
first highlighted the skull rim more often, the second gave practically the
same maps, so the final layer is kept.

**Out-of-distribution check.** `train.py` saves the feature vectors of the
training scans (`ood_reference.npz`) and a threshold. For every upload the app
computes the cosine distance, in the model's own feature space, to its 5th
nearest training scan; above the threshold it shows "unlike the training
scans" instead of a prediction. The threshold is the 99th percentile of the
training scans' own leave-one-out distances, so about 1% of genuine scans are
expected to be flagged. In a test with a Br35H-only model it flagged 1 of 84
unseen Br35H scans and rejected every non-MRI test image (color and grayscale
photos, a terrain map, a plot, noise), as well as side-view MRIs the model had
never seen.

### 6. Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use synthetic images and an untrained model, so they need neither
the dataset nor a trained model file.

---

## 🧪 Method

1. Images are resized to 224×224 and passed through the backbone's own
   preprocessing (built into EfficientNetV2).
2. The pretrained EfficientNetV2-B0 base extracts general visual features.
3. A small head (`GlobalAveragePooling → Dropout → Dense`) predicts the
   tumor probability.
4. Stage 1 trains only the head; stage 2 fine-tunes the last 30 base layers
   (`FINE_TUNE_LAYERS`) with a very small learning rate.
5. Early stopping and model checkpointing are driven by the **validation** AUC;
   the final numbers are reported on a separate **test** set. Each stage gets
   its own early-stopping callback (Keras does not reset its best score
   between `fit()` calls), while one checkpoint keeps the best model across
   both stages.
6. Duplicate and near-duplicate images are removed first: each image gets a
   64-bit DCT perceptual hash (also of its mirrored, flipped and 180°-rotated
   versions), and an image within
   the distance threshold of an already-kept image is dropped. Copies whose
   labels disagree are dropped entirely. Then the label review is applied.
7. The split is then made once on the list of image files (stratified by
   class), so train, validation and test are guaranteed to be disjoint.
8. The decision threshold is 0.5. Tuning it on the validation set (Youden's J,
   `TUNE_THRESHOLD` in `config.py`) is implemented but **off**, because
   cross-validation showed it does not help (see Results).
9. Three models are trained: the first on the standard train/val split, the
   others on the same pool (train + val) with different random validation
   splits and seeds. The app averages their probabilities.

---

## 📈 Results

All numbers are on **distinct scans** (after duplicate removal and label
review), threshold 0.5. Cross-validation folds are made per source, so Br35H
uses the same 5 folds in every experiment and results can be compared image by
image.

**Final model: ensemble of 3 × EfficientNetV2-B0 @ 224 px, trained on Br35H +
Sartaj** (2294 scans), nested 5-fold cross-validation (`ensemble.py`):

| Metric | Br35H (416 scans) | Sartaj (1878 scans) |
|---|---|---|
| Accuracy | **85.3% ± 4.0%** | **96.1% ± 1.2%** |
| ROC AUC | **0.931 ± 0.017** | **0.974 ± 0.010** |
| Sensitivity (tumor) | **87.0% ± 2.2%** | **98.2% ± 0.5%** |
| Specificity (healthy) | **82.9% ± 8.1%** | **74.5% ± 12.9%** |

Mean ± sample std over the 5 outer folds. Pooled: on Br35H 33 of 253 tumors
missed and 28 of 163 healthy scans flagged (61 errors); on Sartaj 30 of 1706
tumors missed and 44 of 172 healthy scans flagged (74 errors). Over both
sources together the accuracy is 94.1% ± 1.3%, but that figure is dominated by
Sartaj's many tumor scans and says little about healthy scans; specificity
(false alarms) remains the main weakness. The single members made 60, 61 and
67 errors on Br35H (ensemble: 61), so the ensemble mainly adds stability.
Compared with the Br35H-only ensemble (58 errors on the same 416 scans,
sensitivity 90.9%, specificity 78.6%), the result on Br35H is about the same
with a shift from sensitivity to specificity, while generalisation to a second
source is far better (see below).

**Effect of adding the second dataset** (single model, 5-fold CV, the same 416
Br35H images in every row):

| Training data | Br35H accuracy | AUC | Sensitivity | Specificity | Errors |
|---|---|---|---|---|---|
| Br35H only | 85.3% | 0.919 | **90.1%** | 77.9% | 61 |
| Br35H + Sartaj, plain class weights¹ | 76.5% | 0.884 | 66.0% | **93.1%** | 97 of 413 |
| **Br35H + Sartaj, balanced weights** | **85.3%** | **0.929** | 85.8% | **84.7%** | **61** |

¹ earlier run, before the duplicate fix of cross-label matches (413 images).

- With balanced weighting the second dataset does not cost anything on Br35H
  (same 61 errors: 22 fixed, 22 new) and moves the balance from sensitivity
  towards specificity, the previous main weakness.
- Its real benefit is generalisation: a Br35H-only ensemble tested on Sartaj
  reached only AUC 0.760 and flagged 3 of 4 healthy Sartaj scans as tumor,
  while the combined model reaches AUC 0.975 on Sartaj in cross-validation
  (accuracy 96.2%, sensitivity 98.3%, specificity 75.7%).

**How the backbone and ensemble were chosen** (on Br35H alone, before the
second dataset was added). A single training run is partly luck: with the
same data but another seed and validation split, its accuracy moves by
2–3 points. So each candidate was trained 15 times (5 folds × 3 seeds/splits,
identical for all candidates) and compared on the average:

| Model | Single run, mean of 15 | Errors per run (of 416) | Ensemble of 3 |
|---|---|---|---|
| MobileNetV2 @ 160 px (original) | 82.4%, AUC 0.900 | 63 – 80 | 84.4% (65 errors) |
| **EfficientNetV2-B0 @ 224 px** | **84.4%, AUC 0.925** | 61 – 69 | **86.1% (58 errors)** |

- EfficientNetV2-B0 was better by +1.9 points on average (better in 9 of 15
  paired runs) and in AUC, and it varies less between runs. The gain is
  consistent in direction but **not statistically significant** (paired
  t-test p ≈ 0.16; per-image McNemar test for the ensembles p ≈ 0.43).
- Averaging 3 models adds about +1.7 to +2.0 points over an average single run
  and removes most of the run-to-run luck.
- A higher resolution alone did not help (MobileNetV2 @ 224 vs @ 160 made the
  same number of errors in an earlier comparison).

**How the numbers changed as leaks were fixed** (Br35H). The same evaluation
gave much higher numbers before all copies were removed, which shows how much
duplicates inflate results on this dataset:

| Data used for evaluation | Distinct images | Best model, CV accuracy |
|---|---|---|
| Copies matched left-right only | 622 | 89.1% (inflated) |
| Upside-down / rotated copies also removed, label review applied | **416** | **86.1%** |

The original code did not deduplicate at all, so any score it reported would
have been inflated even more.

**Threshold tuned on the validation set instead of 0.5:** on Br35H alone the
tuned threshold ended up at 0.48–0.52 and gave the same accuracy; with both
datasets it moved to 0.58–0.69 and gave slightly lower accuracy (93.5% vs
94.3% for a single model over both sources), trading sensitivity for
specificity. The default stays at 0.5.

**Suspected label issues:** with both datasets, 10 images are confidently
contradicted by the model (out-of-fold P ≥ 0.9 against the label), 9 of them
from Sartaj. Most look like model errors on unusual slices (low slices through
the eyes or skull base, other sequences) or on clearly visible tumors the model
missed, rather than wrong labels; a sagittal "glioma" image without a visible
lesion and Br35H `y20` are the least clear. They are listed in the review
sheet for an expert; none was excluded.

**Single train/val/test split** (`train.py`, the ensemble used by the app,
230 test images): accuracy 92.6%, ROC AUC 0.950, sensitivity 95.9%
(189/197), specificity 72.7% (24/33). Per source: Br35H 78.6% (42 images),
Sartaj 95.7% (188 images). With only 16 healthy Br35H test scans, a single
false alarm moves Br35H specificity by 6 points, so the cross-validated
numbers above are the ones to trust. The app shows those when they match the
deployed model.

---

<a id="limitations"></a>

## ⚠️ Limitations & disclaimer

- The model is trained on two public datasets (~2300 distinct scans, ~15%
  healthy), not on clinical data. A model trained on one of them generalised
  poorly to the other, so performance on scans from other hospitals,
  scanners or sequences is unknown. The app's feature-space check warns about
  images unlike the training scans, but it cannot guarantee correctness.
- Labels come from the datasets and are not verified; the review sheet lists
  doubtful ones for an expert.
- **Grad-CAM is only a rough hint.** Training on both datasets moved the
  heatmap closer to the tumor on several Br35H scans, but on some it still
  highlights the centre of the brain. Higher-resolution layers and Grad-CAM++
  did not help. Do not read it as a tumor outline.
- This project is a **decision-support / educational tool** and is **not** a
  substitute for a qualified physician. Final diagnosis must be made by a
  radiologist.

---

## 📄 License

Released under the [MIT License](LICENSE).
