# Stellar Classification Models

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![UNIFEI](https://img.shields.io/badge/UNIFEI-ECOM08A-red)

A comprehensive machine learning project implementing and comparing multiple classification models for astronomical object classification using the Sloan Digital Sky Survey (SDSS17) dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [Models Implemented](#models-implemented)
- [Results Comparison](#results-comparison)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Autorship Note](#autorship-note)
- [License](#license)

## 🔭 Overview

This project builds and evaluates four distinct machine learning classification models to classify astronomical objects from the SDSS17 survey into three categories:
- **Stars** ⭐
- **Galaxies** 🌌
- **Quasars** ✨

The project demonstrates the end-to-end machine learning pipeline: data exploration, preprocessing, model training, evaluation, and comparative analysis.

## 📊 Dataset

**Source:** [Kaggle - Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

### Dataset Characteristics

- **Total Samples:** 100,000 astronomical observations
- **Total Features:** 17 attributes
- **Classes:** 3 (Galaxy, Star, Quasar)

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `obj_ID` | ID | Object identifier in the image catalog |
| `alpha` | Angle | Right ascension (longitude projected on the sky) |
| `delta` | Angle | Declination (latitude projected on the sky) |
| `u` | Magnitude | Ultraviolet filter magnitude |
| `g` | Magnitude | Green spectrum filter magnitude |
| `r` | Magnitude | Red spectrum filter magnitude |
| `i` | Magnitude | Near-infrared filter magnitude |
| `z` | Magnitude | Infrared filter magnitude |
| `run_ID` | ID | Telescope scan identifier |
| `rerun_ID` | ID | Image reprocessing number |
| `cam_col` | ID | Camera column identifier |
| `field_ID` | ID | Sky field identifier |
| `spec_obj_ID` | ID | Object identifier in spectroscopy catalog |
| `class` | Target | Classification (Galaxy, Star, Quasar) |
| `redshift` | Physical | Cosmological redshift (distance/velocity indicator) |
| `plate` | ID | Aluminum plate identifier |
| `MJD` | Date | Modified Julian Date of observation |
| `fiber_ID` | ID | Optical fiber identifier |

## 🔧 Data Processing

### 1. Data Cleaning
- Identified and removed artificial values (-9999) in magnitude columns (u, g, z)
- Removed incomplete or erroneous telescope measurements
- Final dataset: **~96,000 samples** (after cleaning)

### 2. Outlier Detection
- Applied **DBSCAN clustering** algorithm for outlier removal
- Parameters: eps=0.75, min_samples=16
- Outliers removed: ~15% of data
- Final clean dataset: **~81,000 samples**

### 3. Feature Engineering
- Removed non-informative columns: identification numbers and technical telescope parameters
- Used **Mutual Information (MI)** analysis to rank feature importance
- Final features for modeling: 8 features (alpha, delta, u, g, r, i, z, redshift)

### 4. Data Standardization
- Applied StandardScaler normalization (zero mean, unit variance)
- Train/Test Split: 80/20 with stratification to maintain class balance

### 5. Label Encoding
- Encoded class labels to numerical values:
  - 0 = Galaxy
  - 1 = Star
  - 2 = Quasar

## 🧠 Models Implemented

### 1. Logistic Regression
**Approach:** Linear probabilistic classifier using Softmax activation for multiclass problems

**Hyperparameters:**
- Multi-class: Multinomial
- Solver: LBFGS
- Max iterations: 1000

**Results:**
- Accuracy: **96%**
- Best-performing class: Star (97% F1-Score)
- Worst-performing class: Quasar (91% F1-Score)

**Analysis:** Shows that data has a linear separability component but struggles with overlapping regions between Galaxies and Quasars.

---

### 2. Support Vector Machine (SVM)
**Approach:** Kernel-based method using Radial Basis Function (RBF) for non-linear separation

**Hyperparameters:**
- Kernel: RBF (Radial Basis Function)
- C (regularization): 60
- Gamma: Scale (automatic)
- Random state: 42

**Results:**
- Accuracy: **97%**
- Precision & Recall: Balanced across all classes
- Significant improvement over Logistic Regression

**Analysis:** RBF kernel effectively handles non-linear decision boundaries, capturing the overlapping patterns better than linear methods.

---

### 3. Multi-Layer Perceptron (MLP)
**Approach:** Artificial neural network for learning complex, non-linear relationships

**Hyperparameters:**
- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.001
- Hidden layers: 1 × 100 neurons
- Max iterations: 200

**Results:**
- Accuracy: **97%**
- Training accuracy: 97.40%
- Test accuracy: 97.37%
- No significant overfitting detected

**Analysis:** Neural network converges rapidly (within 20 iterations) while maintaining excellent generalization on unseen data.

---

### 4. Decision Tree
**Approach:** Recursive partitioning using information gain-based tree construction

**Hyperparameters:**
- Criterion: Gini index
- Max depth: 8
- Random state: 42

**Results:**
- Accuracy: **98%** ⭐ (Best performer)
- Galaxy class: 98% F1-Score
- Quasar class: 94% F1-Score
- Star class: 100% F1-Score

**Analysis:** Achieves best overall performance through interpretable decision rules. Feature importance ranking reveals **redshift** as the primary discriminator.

## 📈 Results Comparison

### Performance Metrics Summary

| Model | Accuracy | Galaxy F1 | Quasar F1 | Star F1 | Computational Cost |
|-------|----------|-----------|-----------|---------|-----------------|
| **Logistic Regression** | 96% | 96% | 91% | 97% | Very Low |
| **SVM (RBF)** | 97% | 97% | 94% | 98% | Medium |
| **MLP** | 97% | 98% | 95% | 98% | High |
| **Decision Tree** | **98%** | **98%** | **94%** | **100%** | Low |

### Confusion Matrix Analysis

**Key Observations:**
1. **Star classification:** Nearly perfect (99-100% recall across all models)
2. **Galaxy classification:** Consistently accurate (95-99% recall)
3. **Quasar classification:** Most challenging across all models (89-94% recall)

**Quasar Misclassification Pattern:**
- Quasars misclassified as Galaxies: ~11-12% (primary error source)
- Quasars misclassified as Stars: <1% (rare)

## 🔬 Key Findings

### Feature Importance

**Redshift is the dominant discriminator:**
- Mutual Information (MI) score: ~0.80 (highest among all features)
- Provides clear separation between classes due to cosmological distance-velocity relationship
- Nearly perfect separation for Stars (very low redshift values)
- Substantial overlap between Galaxies and Quasars at certain redshift ranges

**Secondary important features (in order):**
1. z (infrared magnitude)
2. i (near-infrared magnitude)
3. r (red magnitude)
4. u (ultraviolet magnitude)
5. g (green magnitude)

**Least important features:**
- alpha, delta (sky coordinates - no physical relevance to classification)

### Why Quasars Are Hard to Classify

1. **Physical similarity:** Quasars are active galactic nuclei (AGN) - essentially galaxies with supermassive black holes
2. **Photometric overlap:** Color patterns (u, g, r, i, z magnitudes) overlap significantly with normal galaxies
3. **Redshift ambiguity:** Some galaxies have redshift values in the same range as nearby quasars
4. **Data outliers:** Galaxy outliers at high redshift values create classification ambiguity

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Required Libraries

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 👥 Authorship Note
This project was originally developed as a group assignment with **Letícia Borges**. This specific repository is maintained by **Maria Clara Rodrigues Ribeiro** and focuses exclusively on the **Classification Models** and the comparative analysis of their results.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Dataset: [Kaggle - Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
- SDSS Survey: [Sloan Digital Sky Survey](https://www.sdss.org/)
- Scikit-learn Documentation: [sklearn.org](https://scikit-learn.org/)

---
*Developed as a final assignment for the Artificial Inteligence course at UNIFEI.*