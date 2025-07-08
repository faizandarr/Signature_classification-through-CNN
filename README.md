# Signature Recognition and Classification using CNN

This project involves building a complete pipeline to process scanned handwritten signatures, extract and organize them, and classify them using Convolutional Neural Networks (CNNs). It also includes performance comparison with traditional feature engineering techniques such as HOG and SIFT.

## 📌 Problem Statement

Given a set of images containing handwritten signatures arranged in tabular form, the goal is to:
- Detect and extract individual signatures.
- Segment and organize them into folders based on individual IDs.
- Perform train-test split for classification.
- Train a CNN model to classify signatures to their corresponding individuals.
- Compare CNN-based feature extraction with manual techniques like HOG and SIFT.
- Evaluate model performance using accuracy, precision, recall, and F1-score.

---

## 🛠 Techniques & Tools Used

- **Language:** Python
- **Libraries:** OpenCV, PyTorch, NumPy, Matplotlib, Scikit-learn
- **Models:** Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN)
- **Feature Engineering:** HOG (Histogram of Oriented Gradients), SIFT (Scale-Invariant Feature Transform)

---

## 🧠 Core Workflow

### 1. **Image Preprocessing**
- **Edge & Contour Detection:** Applied Canny and hierarchical contour detection to localize signatures.
- **Thresholding & Morphology:** Adaptive thresholding and closing operations to clean images.
- **Clustering & Filtering:** Used clustering and area-based filtering to isolate and crop signature regions.
- **Rotation & Padding:** Normalized dataset by rotating and augmenting images to balance sample sizes.

### 2. **Data Organization**
- Extracted signature patches saved in person-specific folders.
- Ensured consistent dataset structure for training and validation.

### 3. **Model Training**
- Implemented CNN model using PyTorch with configurable image size, batch size, and epochs.
- Compared performance with a baseline ANN model.
- Plotted training and validation accuracy/loss graphs.

### 4. **Evaluation**
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- CNN significantly outperformed ANN in both accuracy and loss reduction over epochs.

---

## 📊 Results Snapshot

| Epoch | CNN Accuracy | ANN Accuracy |
|-------|--------------|--------------|
| 1     | 33.26%       | 0.22%        |
| 5     | 54.57%       | 0.87%        |
| 10    | 61.24%       | 0.87%        |

- CNN achieved consistent improvement and learned effective spatial features.
- ANN struggled with convergence and underperformed.

---

## 🔍 Future Improvements

- Apply OCR to automatically link signatures to textual IDs.
- Introduce more advanced CNN architectures (e.g., ResNet).
- Experiment with transformer-based models for signature classification.
- Improve imbalance handling with SMOTE or advanced augmentation.

---

## 📑 Report

See [Signatures_Classification.pdf](./Signatures_Classification.pdf) for full technical documentation, method comparisons, and visualizations.

---

## 👨‍💻 Author

**Mohammad Faizan**  
[LinkedIn](#) | [GitHub](#) | [Email](#)

---
