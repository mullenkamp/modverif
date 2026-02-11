# model_eval Roadmap & TODO

This document outlines the current state of meteorological verification metrics in `model_eval` and prioritizes future development, aligning with standards set by the MET (Model Evaluation Tools) framework.

## ✅ Completed Metrics

### Continuous (CNT)
Standard metrics for continuous fields like Temperature, Pressure, and Wind.
- [x] **Normalised Error (NE)** - Percentage error relative to source.
- [x] **Absolute Normalised Error (ANE)** - Absolute percentage error.
- [x] **Root Squared Error (RSE)** - Element-wise RMSE.
- [x] **Root Mean Square Error (RMSE)** - Domain-aggregated squared error.
- [x] **Mean Error (Bias)** - Systematic over/under-prediction.
- [x] **Pearson Correlation** - Linear correlation between model and observations.

### Categorical (CTS)
Binary "Yes/No" verification based on a threshold (e.g., Precip > 1.0mm).
- [x] **Contingency Table Engine** - Calculation of Hits (A), False Alarms (B), Misses (C), and Correct Negatives (D).
- [x] **Probability of Detection (POD)** - Hit Rate.
- [x] **False Alarm Ratio (FAR)** - Ratio of false alarms to total forecasts.
- [x] **Critical Success Index (CSI)** - Threat Score.
- [x] **Gilbert Skill Score (GSS)** - Equitable Threat Score (accounts for chance hits).
- [x] **Frequency Bias (FBIAS)** - Ratio of forecast events to actual events.

---

## 🚀 High Priority (Short Term)

### 1. High-Level API Integration
- [ ] **Threshold Parameter:** Update `evaluate_models_cell` and `evaluate_models_domain` to accept a `threshold` argument.
- [ ] **Categorical Branching:** Update orchestrators in `evaluate.py` to handle `ContingencyTable` metrics when a threshold is provided.
- [ ] **Metadata Mapping:** Complete `_get_metric_info` in `metrics.py` for all categorical metrics to ensure NetCDF4 outputs have correct units/names.

### 2. Basic Spatial Verification
- [ ] **Fractions Skill Score (FSS):** Implement FSS to evaluate high-resolution rainfall forecasts without the "double penalty" of small spatial displacements.

---

## 📈 Medium Priority (Mid Term)

### 1. Vector Statistics
- [ ] **Vector RMSE:** Specialized RMSE for wind vectors (U/V components).
- [ ] **Wind Speed Bias:** Standard bias specifically for magnitude.

### 2. Time Series Enhancements
- [ ] **Lagged Correlation:** Check for timing offsets in model features (e.g., a storm arriving 2 hours early).
- [ ] **Diurnal Cycle Analysis:** Metrics aggregated by hour-of-day to identify solar radiation/heating biases.

---

## 🔬 Long Priority (Future)

### 1. Advanced Spatial Methods
- [ ] **Object-Based Verification:** A simplified version of MET's MODE (identifying "blobs" of rain and comparing their area, centroid, and orientation).
- [ ] **Neighborhood MAE/RMSE:** Evaluation within a search radius.

### 2. Probabilistic Verification
- [ ] **Brier Score:** If the package expands to support ensemble model outputs.
- [ ] **Reliability Diagrams:** Visualization tool for probabilistic forecasts.

### 3. Visualization
- [ ] **Performance Diagrams:** Plot POD vs (1-FAR) to visualize categorical skill.
- [ ] **Taylor Diagrams:** Combine Correlation, RMSD, and Standard Deviation in one plot.
