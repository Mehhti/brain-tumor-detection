# 🧠 Brain Tumor Detection with Deep Learning

Detect brain tumors from MRI images using a two-stage **transfer learning**
model (MobileNetV2), served through a simple **Streamlit** web app.

The project covers the full workflow: data loading, model training with a
proper train/validation/**independent-test** split, honest evaluation, and an
interactive app that also validates whether the uploaded image is actually a
brain-MRI scan.

---

## ✨ Features

- **Transfer learning** with MobileNetV2 (pretrained on ImageNet).
- **Two-stage training:** train the classifier head, then fine-tune the top layers.
- **Standard 3-way split** (train / validation / test) so the reported score is
  measured on data the model never saw.
- **Class weighting** for robustness against class imbalance.
- **Input validation** in the web app: rejects non-MRI images (color photos,
  documents) and flags low-confidence predictions.
- Clean, configurable code (all settings live in `config.py`).

---

## 📁 Project structure

```
brain-tumor-detection/
├── app.py             # Streamlit web app
├── train.py           # Model training script
├── image_check.py     # Input-image suitability check
├── config.py          # Central configuration (paths & hyperparameters)
├── requirements.txt   # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

> Note: the dataset and the trained model file are **not** included in the repo
> (see `.gitignore`). Download the dataset and train the model locally, or place
> a pretrained model file next to `app.py`.

---

## 📊 Dataset

This project uses the **Br35H :: Brain Tumor Detection 2020** dataset
(3000 balanced images: 1500 tumorous, 1500 non-tumorous).

- Download: <https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection>

After extracting, point `DATA_DIR` (in `config.py`) at the folder that directly
contains the `yes/` and `no/` subfolders:

```
data/
├── yes/    # tumorous MRI images
├── no/     # non-tumorous MRI images
└── pred/   # optional, ignored by the code
```

---

## 🚀 Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

### 2. Configure the data path

Edit `config.py` (or set an environment variable):

```bash
export DATA_DIR="/path/to/brain_tumor_dataset"
```

### 3. Train the model

```bash
python train.py
```

This trains the model, prints a classification report on the **independent test
set**, saves the best model to `best_model_br35h.keras`, and writes a training
curve to `training_history.png`.

### 4. Run the web app

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal (usually
`http://localhost:8501`) and upload an MRI image.

---

## 🧪 Method

1. Images are resized to 160×160 and passed through MobileNetV2's own
   preprocessing.
2. The pretrained MobileNetV2 base extracts general visual features.
3. A small head (`GlobalAveragePooling → Dropout → Dense`) predicts the
   tumor probability.
4. Stage 1 trains only the head; stage 2 fine-tunes the last 30 base layers
   with a very small learning rate.
5. Early stopping and model checkpointing are driven by the **validation** AUC;
   the final numbers are reported on a separate **test** set.

---

## ⚠️ Limitations & disclaimer

- The model is trained on one dataset. Performance drops on **out-of-distribution**
  images (scans from very different sources, or non-MRI images). For clinical use,
  training on multi-source data is required.
- This project is a **decision-support / educational tool** and is **not** a
  substitute for a qualified physician. Final diagnosis must be made by a
  radiologist.

---

## 📄 License

Released under the [MIT License](LICENSE).
