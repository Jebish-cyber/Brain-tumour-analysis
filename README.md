# Quantum Brain Tumor Analysis & Classification

A Hybrid Classical-Quantum Deep Learning framework for MRI brain tumor classification leveraging **PennyLane** and **TensorFlow / Keras**.

---

## 📌 Project Overview

This repository implements a **Hybrid Quantum-Classical Neural Network (HQNN)** designed to classify brain MRI scans into four diagnostic categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

By combining classical convolutional feature extractors with a parameterized Variational Quantum Circuit (VQC), the model explores quantum-enhanced representation spaces for medical image analysis.

---

## 🧬 Architecture & Methodology

1. **Preprocessing & Augmentation**:
   - Resizing MRI scans to $64 \times 64$ pixels with normalized intensities $[0, 1]$.
   - Batching and caching via `tf.data` pipeline.

2. **Classical Feature Extraction**:
   - Convolutional layers (`Conv2D`, `MaxPooling2D`, `BatchNormalization`) compress spatial visual features into a low-dimensional bottleneck vector.

3. **Quantum Variational Layer**:
   - **4-Qubit Quantum Device** (`default.qubit` simulated via PennyLane).
   - **Angle Embedding**: Input features are encoded using $R_y$ rotation gates.
   - **Parameterized Rotations**: Trainable single-qubit Euler rotations ($R(\phi, \theta, \omega)$).
   - **Entanglement**: Linear/circular CNOT entanglement layers creating quantum correlations.
   - **Measurement**: Expectation values $\langle Z_i \rangle$ measured across all wires.

4. **Classification Head**:
   - Dense layers map expectation values to class logits followed by a 4-class Softmax activation.

---

## 📂 Repository Structure

```text
├── quantum-brain-tumor-prototype.ipynb   # Main Jupyter Notebook with model training & evaluation
├── .gitignore                            # Standard git ignore rules
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pennylane tensorflow matplotlib scikit-learn
```

### Dataset

The pipeline uses the **Brain Tumor MRI Dataset** (Masoud Nickparvar):
- Contains Training and Testing sets across 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary`.
- If running on Kaggle, the notebook automatically configures dataset paths from `/kaggle/input`.

### Running the Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook "quantum-brain-tumor-prototype.ipynb"
```

Execute cells sequentially to inspect:
- Data loading and distribution visualizations
- Quantum circuit definition & gradients via TensorFlow backpropagation
- Hybrid model training with `ReduceLROnPlateau` and `EarlyStopping` callbacks
- Confusion matrix, ROC curves, and sample test inference predictions

---

## 📊 Results & Outputs

The notebook generates:
- **Multi-class Loss & Accuracy curves** comparing training vs validation performance.
- **Confusion Matrix** evaluating precision and recall across tumor types.
- **Single-sample Inference Inspection** displaying model prediction probabilities against ground truth labels.

---

## 🛠️ Tech Stack

- **PennyLane**: Quantum circuit design, parameterized gates, and quantum machine learning interface.
- **TensorFlow / Keras**: Deep learning framework for classical layers and hybrid model orchestration.
- **NumPy & Pandas**: Numerical operations and data structures.
- **Matplotlib & Scikit-learn**: Data visualization and classification evaluation metrics.
