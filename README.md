# 🧬 CNV-Based Parameter Inference with Deep Sets

This project applies deep learning to infer selection coefficients from simulated copy number variation (CNV) data using Deep Set architectures.

## 📌 Overview

We tackle three tasks:

1. **Classification** of simulation regime: Laplace, Gaussian-Low, Gaussian-High  
2. **Regression** to predict the **mean selection coefficient**  
3. **Regression** to predict the **full vector of 44 chromosome-specific selection coefficients**  

---

## 📂 Dataset

Each simulation contains:
- **25 trials** of CNV vectors (length 44)
- A **parameter vector** (length 46), where `parameter[2:]` are selection coefficients

Three types of simulation:
- `numpy_data_laplace/`
- `numpy_data_low/`
- `numpy_data/` (Gaussian-High)

All datasets are balanced to equal size before training.



## 📁 Code Overview

| File                     | Description                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `model.py`               | Defines the **Normalizing Flow** model architecture used for generative modeling tasks.                            |
| `SBI_main.ipynb`         | Apply the **Normalizing Flow** model for generating the parameter distribution.                                    |
| `net_builder.py`         | Implements the **Deep Set** encoder, shared across classification and regression models.                           |
| `classifier.py`          | Trains a neural network to **classify simulated data** based on the underlying selection coefficient distribution. |
| `classifier_KFold.py`    | Performs **K-Fold Cross-Validation** to evaluate classification performance across multiple data splits.           |
| `regression_1param.py`   | Trains a model to **predict the mean** of the selection coefficient vector from simulated CNV data.                |
| `regression_allparam.py` | Trains a model to **predict all 44 selection coefficients** (one per chromosome) from simulated data.              |
