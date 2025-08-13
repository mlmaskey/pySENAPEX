# simAPEX: Calibration, Sensitivity, and Uncertainty Driver

This repository contains the `simAPEX` driver script and related utilities for running **APEX/APEXgraze** simulations in three modes:

- **Calibration** – Generate parameter sets, run APEX, and collect performance metrics (PEM).
- **Sensitivity** – Perturb calibrated parameters to produce datasets for **PBSA** (Percent-Based Sensitivity Analysis) and **SRC** (Standardized Regression Coefficient) analysis.
- **Uncertainty** – Generate ensembles from within-criteria runs to assess prediction uncertainty.

This document describes:

1. **Workflow & Modes** (`simAPEX` runbook)
2. **Inputs, Outputs, and Criteria**
3. **Algorithm for SRC Calculation** (post-processing sensitivity outputs)

---

## 1. Workflow & Modes

### 1.1 Calibration Mode
1. **Read parameter ranges** from `file_limits` in `src_dir/Utility`.
2. **Generate parameter sets**:
   - `isall=True`: sweep all parameters uniformly from min to max.
   - `isall=False`: sweep only sensitive parameters (`read_sensitive_params`).
3. **Run loop** for each simulation:
   - Pick a parameter set (`pick_param`).
   - Write to APEX inputs (`overwrite_param`, `modify_list`).
   - Run `APEXgraze.exe` (native or via Wine).
   - Read outputs: daily, basin, annual.
   - Calculate PEM metrics (COD, NSE, PBIAS, etc.).
   - Save run-specific outputs and parameter set to CSV.

### 1.2 Sensitivity Mode
1. **Load calibration results** (`file_pem`, `file_param`).
2. **Select “best” run** meeting criteria:
   - COD ≥ `COD_criteria`
   - NSE ≥ `NSE_criteria`
   - |PBIAS| ≤ `PBIAS_criteria`
3. **Build sensitivity matrix**:
   - Percent deltas: `np.arange(-max_range, max_range+1, increment)`
   - **Block 1**: All parameters perturbed together.
   - **Block 2**: One parameter varied at a time.
4. **Run simulations** for all parameter sets.
5. **Save outputs** to `OutputSensitivity/`.

### 1.3 Uncertainty Mode
1. **Select within-criteria runs** from calibration.
2. **Generate ensembles** using mean ± std for each parameter (bounded to min/max).
3. **Run simulations** and save outputs to `OutputUncertainty/`.

---

## 2. Inputs & Outputs

### 2.1 Key Inputs
- **Config file** (`runtime.ini`):
  - `run_name`, `file_limits`, `n_discrete`, `n_simulation`, `n_start`
  - Criteria: `COD_criteria`, `NSE_criteria`, `PBIAS_criteria`
  - Sensitivity/uncertainty: `max_range`, `increment`, `max_range_uncertaintity`, `increment_uncertainty`
- **Parameter limits file**: `<src_dir>/Utility/<file_limits>`
- **Calibration results** (for sensitivity/uncertainty):
  - PEM CSV: `dir_calibrate_res/file_pem`
  - Parameter CSV: `dir_calibrate_res/file_param`

### 2.2 Outputs per Run
- **Daily**: `daily_outlet_0000001.csv`, `daily_basin_0000001.csv`
- **Annual**: `annual_0000001.csv`
- **Crop-specific splits**
- **Parameter archives**: `APEXPARM` and `selected_APEXPARM.csv` (sensitivity/uncertainty)

---

## 3. SRC Calculation Algorithm

After **Sensitivity Mode** produces outputs, SRC is calculated in the analysis module (e.g., `senanaAPEX`).  

**Notation:**
- \( n \) = number of sensitivity runs
- \( p \) = number of parameters
- \( X_{i,j} \) = value of parameter \( j \) in run \( i \)
- \( Y_i \) = performance metric for run \( i \) (e.g., NSE)

---

### 3.1 Steps

**Step 1: Load Data**
```python
# X: parameter values (n x p)
# Y: performance metric (n x 1)
X, Y = load_sensitivity_data("OutputSensitivity/")
````

**Step 2: Standardize**

$$
Z_{i,j} = \frac{X_{i,j} - \mu_j}{\sigma_j}, \quad Y'_i = \frac{Y_i - \mu_Y}{\sigma_Y}
$$

Where $\mu_j, \sigma_j$ are mean and std dev of parameter $j$,
and $\mu_Y, \sigma_Y$ are mean and std dev of $Y$.

**Step 3: Regression**

* **First-order SRC** (one-param-at-a-time block):

$$
\beta_j = \frac{\text{Cov}(Z_{\cdot,j}, Y')}{\text{Var}(Z_{\cdot,j})}
$$

* **Total SRC** (all-params block): Fit multiple linear regression:

$$
Y' = \beta_0 + \sum_{j=1}^{p} \beta_j Z_{\cdot,j} + \epsilon
$$

Estimate $\beta_j$ using least squares:

$$
\boldsymbol{\beta} = (Z^\top Z)^{-1} Z^\top Y'
$$

**Step 4: Output**

* SRC values for each parameter.
* Use **absolute SRC** for ranking importance.

---

### 3.2 Interpretation

* **|SRC| close to 1**: Strong influence on metric.
* **Positive SRC**: Parameter increase → metric increase.
* **Negative SRC**: Parameter increase → metric decrease.
* Compare PBSA (direct percent change effects) vs. SRC (combined regression effects).

---

## 4. Minimal Example Usage

```python
from pathlib import Path
from configobj import ConfigObj
from simAPEX import simAPEX
from inAPEX import inAPEX

config = ConfigObj("runtime.ini")
src_dir = Path("/path/to/repo")
wine = "/usr/bin/wine"  # "" if Windows

inp = inAPEX(config, src_dir)

# Calibration
simAPEX(config, src_dir, wine, inp, model_mode="calibration", isall=True)

# Sensitivity
simAPEX(config, src_dir, wine, inp, model_mode="sensitivity", isall=False)

# Uncertainty
simAPEX(config, src_dir, wine, inp, model_mode="uncertainty", isall=False)
```

---

## 5. QA Checklist

* [ ] `runtime.ini` criteria are correct (`COD/NSE/PBIAS`).
* [ ] `file_limits` has header row (param names) and correct ranges.
* [ ] Calibration results exist before running sensitivity/uncertainty.
* [ ] Wine/APEXgraze runs without errors.
* [ ] Output directories are writeable.
* [ ] `is_pesticide` matches model setup.

---

## 6. Notes

* SRC/PBSA **plots** are generated in the analysis phase, not here.
* Sensitivity block structure is important for correct SRC estimation.
* Project-specific conventions (e.g., `i <= 69`) should be documented in config or constants.

---

```
