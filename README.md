# 🌾 Wheat Disease Classification

An end-to-end deep learning project that classifies wheat plant diseases from images using transfer learning. Four CNN architectures were trained, evaluated, and compared. The best-performing model (ResNet50) is deployed through an interactive Streamlit web application.

---

## 📁 Project Structure

```
wheat-disease-classification/
│
├── Dataset_preprocessing/
│   └── DatasetCleaning_wheat.ipynb     # Dataset cleaning and filtering
│
├── model_traing/
│   ├── NewTry_EfficientNetB0.ipynb     # EfficientNetB0 training
│   ├── NewTry_EfficientNetV2B0.ipynb   # EfficientNetV2B0 training
│   ├── NewTry_MobileNetV2.ipynb        # MobileNetV2 training
│   └── NewTry_ResNet50.ipynb           # ResNet50 training
│
├── model_comparison/
│   └── 4ModelsConfusionMatrices_Wheat.ipynb  # Evaluation & comparison of all 4 models
│
├── models/
│   ├── resnet50_balanced_500.h5
│   ├── efficientnetb0_balanced_500.h5
│   ├── efficientnetv2b0_balanced_500.h5
│   └── mobilenetv2_balanced_500.h5
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🌾 About Wheat and the Crop

Wheat (*Triticum aestivum*) is one of the most widely cultivated cereal crops in the world and a staple food for billions of people. It is grown across diverse climates — from the plains of South Asia to the fields of Europe and North America. Wheat is highly susceptible to fungal diseases, which can reduce grain yield by 10–70% depending on the severity and timing of infection.

Early and accurate disease detection is critical for farmers to take timely action, protect their yield, and reduce unnecessary pesticide use.

---

## 🦠 Disease Classes

This project classifies wheat images into **5 categories**:

### 1. 🍂 Crown and Root Rot
A disease complex caused by *Fusarium* species and other soilborne pathogens. Affects the crown (base of the stem) and root system. Symptoms include brown or black lesions at the base of the stem, yellowing of leaves, stunted growth, and sometimes "whiteheads" (heads that die prematurely without filling grain). Drought stress can worsen damage. Difficult to control once established — prevention through seed treatment and crop rotation is key.

### 2. 🌾 Healthy Wheat
No visible signs of disease or stress. Normal green foliage, well-formed tillers, and typical growth characteristics. Note that visual inspection alone cannot guarantee a completely disease-free plant; subsurface infections may not yet show symptoms.

### 3. 🟠 Leaf Rust (Brown Rust)
Caused by the fungus *Puccinia triticina* (historically called *Puccinia recondita*). Appears as small, circular to elliptical orange-brown powdery pustules (uredia) mainly on the upper surfaces of leaves and leaf sheaths. Heavy infection reduces photosynthetic area, leading to lower grain weight and yield. Spreads rapidly by windborne spores in warm, moist conditions.

### 4. 🟡 Septoria (Septoria Tritici Blotch / Septoria Nodorum Blotch)
A group of fungal diseases. *Septoria tritici* blotch is caused by *Zymoseptoria tritici*; Septoria nodorum blotch is caused by *Parastagonospora nodorum*. Symptoms appear as tan or brown irregular leaf blotches, often containing tiny black dots called pycnidia (spore-producing structures). Prolonged wet, humid weather favors infection. Can cause serious yield losses if the flag leaf is heavily affected.

### 5. 🔴 Stripe Rust (Yellow Rust)
Caused by *Puccinia striiformis* f. sp. *tritici*. Characterized by narrow yellow to orange-yellow stripes of powdery pustules running parallel to the leaf veins. Pustules may also appear on leaf sheaths, glumes, and necks. Thrives in cool, moist conditions. Severe infections can significantly reduce grain yield. One of the most economically damaging wheat diseases globally.

---

## 1. 🗃️ Dataset Description and Cleaning

### Source
The raw dataset consists of wheat leaf and plant images organized into per-class folders under `Wheat_Images/`:
```
Wheat_Images/
├── Crown and Root Rot/
├── Healthy Wheat/
├── Leaf Rust/
├── Septoria/
└── Stripe Rust/
```

### Cleaning Steps (`DatasetCleaning_wheat.ipynb`)

The dataset was carefully cleaned before training to ensure image quality:

1. **Human Presence Detection** — A Haar cascade face detector (OpenCV) was used to scan all images and flag those containing human faces. Images with detected human presence were identified and removed to avoid introducing irrelevant features into the model.

2. **Watermark / Text Removal** — Some images (particularly Leaf Rust) contained embedded text watermarks or labels at the bottom edge. A bottom-crop operation removed the bottom **12%** of such images to eliminate this noise:
   ```python
   crop_height = int(h * (1 - 0.12))
   cropped_img = img[:crop_height, :, :]
   ```

3. **Format Standardization** — All images were confirmed to be in `.jpg`, `.jpeg`, or `.png` format.

---

## 2. ⚖️ Dataset Balancing and Split

### Balancing
The original dataset had unequal numbers of images per class. To avoid class imbalance during training, exactly **500 images per class** were selected:

- If a class had **≥ 500 images** → random sample **without replacement**
- If a class had **< 500 images** → sample **with replacement** (oversampling) to reach 500

This produced a perfectly balanced dataset of **2,500 images total** (500 × 5 classes).

### Stratified Split
The balanced dataset was then split using **stratified splitting** (via `sklearn.model_selection.train_test_split`) to ensure equal class proportions in every set:

| Split      | Images | Percentage |
|------------|--------|------------|
| Train      | 2,000  | 80%        |
| Validation | 250    | 10%        |
| Test       | 250    | 10%        |

- `random_state=42` was used for reproducibility
- Splits were saved as `.npy` files so all four models trained on the **exact same split**

```
Train: 2000  |  Val: 250  |  Test: 250
Per-class (each split): 400 train / 50 val / 50 test
```

---

## 3. 🖼️ Image Preprocessing

Two separate preprocessing pipelines were used — one for **training** (with augmentation) and one for **validation/test/inference** (deterministic).

### Training Pipeline (with Augmentation)
Applied using `tf.data` for efficient GPU-parallel loading:

1. Read image from disk → decode to `float32` tensor
2. Resize to **244×244**
3. **Random crop** to 224×224 (introduces slight position variation)
4. **Random horizontal flip**
5. **Random brightness** (max delta ±0.08)
6. **Random contrast** (factor 0.9–1.1)
7. **Random saturation** (factor 0.9–1.1)
8. **Random hue** (max delta ±0.02)
9. Clip values to [0, 255]
10. **Model-specific `preprocess_input`** applied inside the model as a Lambda layer

### Validation / Test / Inference Pipeline
1. Resize with padding to **224×224** (preserving aspect ratio) — `tf.image.resize_with_pad`
2. Model-specific `preprocess_input` applied inside the model

### Inference Preprocessing (Streamlit App)
For the deployed app, preprocessing is done with PIL to match the training pipeline exactly:

```python
img = ImageOps.exif_transpose(pil_image)   # correct camera orientation
img = img.convert("RGB")                    # ensure 3 channels
w, h = img.size
m = min(w, h)
img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))  # center square crop
img = img.resize((224, 224), Image.Resampling.LANCZOS)       # resize to model input
```

---

## 4. 🧠 Model Training

All four models were trained on **Google Colab** with a **T4 GPU** using **TensorFlow / Keras**.

### Architecture (Same for All Models)

Each model follows the same transfer learning architecture:

```
Input (224×224×3)
    ↓
Lambda layer: model-specific preprocess_input
    ↓
Pretrained base (ImageNet weights, top removed)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.25)
    ↓
Dense(5, activation='softmax')   ← output: 5 classes
```

### Training Strategy: Two-Phase Transfer Learning

**Phase 1 — Head Training (base frozen)**
- Freeze all base model layers
- Train only the custom classification head
- Optimizer: `Adam(lr=1e-4)`
- Epochs: up to 10
- Callbacks: `EarlyStopping(patience=7)`, `ReduceLROnPlateau(patience=3, factor=0.3)`

**Phase 2 — Fine-Tuning (top 20% of base unfrozen)**
- Unfreeze the last 20% of base model layers
- Lower learning rate to `Adam(lr=1e-5)` to avoid destroying pretrained weights
- Epochs: up to 20
- Same callbacks applied

### Common Hyperparameters

| Parameter         | Value                          |
|-------------------|-------------------------------|
| Image size        | 224 × 224                     |
| Batch size        | 32                            |
| Loss function     | Sparse Categorical Crossentropy |
| Phase 1 LR        | 1e-4                          |
| Phase 2 LR        | 1e-5                          |
| Max head epochs   | 10                            |
| Max fine-tune epochs | 20                         |
| Early stop patience | 7 epochs (val_loss)         |
| LR reduce patience | 3 epochs (factor 0.3)        |
| Dropout           | 0.3 + 0.25                   |
| Unfreeze %        | 20% of base layers            |

---

## 5. 📊 Model Performance

All models were evaluated on the same held-out test set of **250 images (50 per class)**.

### Test Accuracy Comparison

| Model           | Test Accuracy | TTA Accuracy | Layers Unfrozen |
|-----------------|--------------|--------------|-----------------|
| **ResNet50**    | **91.6%**    | **92.0%**    | 35 / 175        |
| EfficientNetV2B0 | 91.2%       | —            | —               |
| MobileNetV2     | 88.4%        | —            | —               |
| EfficientNetB0  | 80.4%        | —            | 47 / 238        |

### ResNet50 Detailed Classification Report (Test Set)

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Crown and Root Rot | 0.90      | 0.92   | 0.91     | 50      |
| Healthy Wheat      | 0.87      | 0.82   | 0.85     | 50      |
| Leaf Rust          | 0.90      | 0.88   | 0.89     | 50      |
| Septoria           | 0.93      | 1.00   | 0.96     | 50      |
| Stripe Rust        | 0.98      | 0.96   | 0.97     | 50      |
| **Overall**        | **0.92**  | **0.92**| **0.92** | **250** |

### MobileNetV2 Detailed Classification Report (Test Set)

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Crown and Root Rot | 0.93      | 0.82   | 0.87     | 50      |
| Healthy Wheat      | 0.84      | 0.84   | 0.84     | 50      |
| Leaf Rust          | 0.81      | 0.86   | 0.83     | 50      |
| Septoria           | 0.90      | 0.92   | 0.91     | 50      |
| Stripe Rust        | 0.94      | 0.98   | 0.96     | 50      |
| **Overall**        | **0.89**  | **0.88**| **0.88** | **250** |

### EfficientNetV2B0 Detailed Classification Report (Test Set)

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Crown and Root Rot | 0.88      | 0.90   | 0.89     | 50      |
| Healthy Wheat      | 0.87      | 0.82   | 0.85     | 50      |
| Leaf Rust          | 0.85      | 0.88   | 0.86     | 50      |
| Septoria           | 1.00      | 0.98   | 0.99     | 50      |
| Stripe Rust        | 0.96      | 0.98   | 0.97     | 50      |
| **Overall**        | **0.91**  | **0.91**| **0.91** | **250** |

### EfficientNetB0 Detailed Classification Report (Test Set)

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Crown and Root Rot | 0.73      | 0.80   | 0.76     | 50      |
| Healthy Wheat      | 0.73      | 0.82   | 0.77     | 50      |
| Leaf Rust          | 0.72      | 0.68   | 0.70     | 50      |
| Septoria           | 0.96      | 0.90   | 0.93     | 50      |
| Stripe Rust        | 0.91      | 0.82   | 0.86     | 50      |
| **Overall**        | **0.81**  | **0.80**| **0.81** | **250** |

### TTA (Test Time Augmentation) — ResNet50

TTA was applied exclusively to ResNet50. Instead of predicting on a single image, **7 augmented versions** of each test image are generated, predictions are averaged, and the final class is taken from the mean:

| Augmentation         | Description                          |
|----------------------|--------------------------------------|
| Original             | No modification                      |
| Horizontal flip      | Mirror left-right                    |
| Brightness +0.05     | Slightly brighter                    |
| Brightness -0.05     | Slightly darker                      |
| Contrast ×1.1        | Higher contrast                      |
| Contrast ×0.9        | Lower contrast                       |
| Central crop (90%)   | Slight zoom-in, resized to 224×224   |

**Result: ResNet50 TTA accuracy = 92.0%** (vs 91.6% without TTA)

---

## 6. 🚀 Streamlit Deployment

The best model (ResNet50) is deployed as an interactive web application using **Streamlit**.

### Features

- **Image Upload** — Upload any `.jpg`, `.jpeg`, or `.png` wheat image
- **Real-time Prediction** — Model runs inference instantly on upload
- **Prediction Card** — Shows the predicted disease with confidence score
- **Disease Description** — Detailed information about the detected condition
- **Recommended Treatment** — Step-by-step treatment guidance
- **Precautions** — Preventive measures to avoid future outbreaks
- **Confidence Breakdown** — Visual bar chart showing prediction probability for all 5 classes
- **Model Comparison Table** — Expandable section showing all 4 model accuracies
- **Disclaimer** — Reminds users to confirm with an agricultural expert

### Installation and Running

**Step 1 — Clone the repository:**
```bash
git clone <repo-url>
cd wheat-disease-classification
```

**Step 2 — Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS / Kali
venv\Scripts\activate           # Windows
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Download model files:**

The trained model files are large (~400MB total) and not included in the repository. You have several options to obtain them:

**Option 1: Contact the repository owner**
- Request the model files directly from the repository maintainer
- They can provide a download link (Google Drive, Dropbox, etc.)

**Option 2: Train your own models**
- Use the training notebooks in the `model_traing/` directory
- Follow the training process with your own dataset
- This requires a GPU (recommended) and substantial time

**Option 3: GitHub Releases (if available)**
- Check if the repository has [GitHub Releases](https://github.com/sahilghogare/Wheat_Disease_Classification/releases)
- If the maintainer has uploaded models there, download them

Place the downloaded model files in the `models/` directory:
```
models/
├── resnet50_balanced_500.h5          (optional but recommended)
├── efficientnetv2b0_balanced_500.h5    (optional)
├── mobilenetv2_balanced_500.h5        (optional)
└── efficientnetb0_balanced_500.h5     (optional)
```

**Note:** At minimum, you need `resnet50_balanced_500.h5` to run the app with the default ResNet50 model. The other models are optional but enable ensemble voting and model comparison features.

**Step 5 — Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Dependencies

```
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.23.0
Pillow>=9.0.0
```

---

## ⚠️ Disclaimer

Disease identification from an image is **preliminary**. Symptoms of different wheat diseases can overlap visually. Always confirm uncertain diagnoses with a qualified agricultural expert or plant pathologist. Follow current local pesticide registration and product-label instructions before applying any treatment.

---

## 🛠️ Technologies Used

| Tool / Library  | Purpose                          |
|-----------------|----------------------------------|
| TensorFlow / Keras | Model building and training   |
| Scikit-learn    | Dataset splitting, metrics       |
| Pillow (PIL)    | Image loading and preprocessing  |
| NumPy           | Numerical operations             |
| OpenCV          | Human presence detection (cleaning) |
| Streamlit       | Web application deployment       |
| Google Colab    | Training environment (T4 GPU)    |
