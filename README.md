# pySENAPEX — User Manual

## 1. Overview
`pysenAPEX` is a Python class for **post-calibration sensitivity analysis** of APEX/APEXgraze model runs.  
This streamlined version supports:
- **PBSA** (percent-based sensitivity analysis) heatmaps
- **SRC** (Standardized Regression Coefficient) analysis
- No SOBOL or FAST indices

Outputs include:
- Parameter change vs. metric heatmaps
- SRC bar plots (first-order and total)
- CSV tables of metrics, parameter values, and statistical summaries
- Bulk export of model output variables (daily/annual)

---

## 2. Requirements

### Python
- Python 3.8 or higher
- The following Python packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `configobj`

### Utility Modules
These must be available in your `Utility/` directory:
- `pyAPEXpost` (with `print_progress_bar`, `pyAPEXpost` class, `get_measure`, `get_stats`, etc.)
- `apex_utility` (with `read_sensitive_params`, `split_data`)
- `easypy` (with `get_outlier_na`, `find_change_percent`, `corr_test`)

---

## 3. Input Data & Configuration

### Configuration File
A `runtime.ini` file in the working directory should define:
```

dir\_sensitivity = path/to/sensitivity/run/folder
file\_limits = filename.csv
Site = site\_name
Scenario = scenario\_name
COD\_criteria = 0.5
NSE\_criteria = 0.5
PBIAS\_criteria = 25
max\_range = 5
warm\_years = 3
calib\_years = 5

````
> **Note:** `PBIAS_criteria` may also be spelled `PBAIS_criteria`; both are supported.

### Required Files
- **Limits file**: `<src_dir>/Utility/<file_limits>` — CSV with parameter bounds
- **Observed data**: `Program/calibration_data.csv`
- **APEX output statistics**:
  - `Statistics_runoff.csv` (for attribute `WYLD`)
  - `Statistics_sediment.csv` (for attribute `YSD`)
  - `Statistics_<attribute>.csv` (for other attributes)
- **APEX parameter file**: `APEXPARM.csv` in the `dir_sensitivity` folder
- **Daily/annual output CSVs**: one per run, e.g.:
  - `daily_outlet_0000001.csv.csv`
  - `daily_basin_0000001.csv.csv`
  - `annual_0000001.csv.csv`

---

## 4. How to Run

```python
from pathlib import Path
from configobj import ConfigObj
from senanaAPEX import senanaAPEX  # your class file

# Load config
config = ConfigObj('runtime.ini')

# Define paths
src_dir = Path("/path/to/src_dir")
out_dir = Path("/path/to/output_folder")
attribute = "WYLD"  # or "YSD", etc.

# Instantiate and run
sa = senanaAPEX(src_dir, config, out_dir, attribute, metric='OF')
````

> The constructor runs the entire analysis and writes outputs to `out_dir`.

---

## 5. Outputs

### Sensitivity Analysis

For each metric (`OF`, `NSE`, `PBIAS`, `COD`):

* **Heatmap PNG**: `Heatmap_<metric>.png`
* **Heatmap CSV**: `Heatmap-data_<metric>.csv`
* **SRC Bar Plot PNG**: `Index_<metric>.png`
* **Parameter Values CSV**: `Param_<metric>.csv`
* **SRC First-order CSV**: `Index_first_<metric>.csv`
* **SRC Total CSV**: `Index_Total_<metric>.csv`
* **Statistical Summary CSV**: `Stat_Summary_<metric>.csv`
* **Criteria Counts CSV**: `Param_within_<metric>.csv`

### Bulk Output Exports

Daily and annual CSVs for selected variables, e.g.:

* `Daily_WYLD.csv`
* `Daily_ET.csv`
* `Annual_YLDG.csv`
* `Annual_BIOM.csv`
  (each includes a `Stage` column: Calibration/Validation)

---

## 6. Interpreting Results

### Heatmaps

* Rows: % change in parameter value (relative to calibrated value)
* Columns: Parameters
* Cell color: % change in performance metric (relative to best run)

### SRC Bar Plots

* **First-order**: direct effect of parameter on metric
* **Total**: direct + interaction effects
* **Log-scale** on x-axis to compare small vs. large sensitivities

---

## 7. Notes & Tips

* **Performance**: Large numbers of runs/parameters will slow down analysis.
* **Parameter Labels**: `PARAM [x]` uses `PARAM_OFFSET = 70` for display; adjust if needed.
* **Criteria Thresholds**: Used for filtering “good” runs in summaries.
* **Output Organization**: Each metric’s outputs are grouped by filename pattern.

---

## 8. Troubleshooting

* **KeyError**: Check that `runtime.ini` has all required keys.
* **FileNotFoundError**: Verify all required input files exist in the expected directories.
* **Blank Heatmap**: Ensure `pe_best[metric]` is non-zero to avoid division by zero.

  Here’s the **SRC computation user manual** in Markdown format, with plotting completely omitted.


# Standardized Regression Coefficient (SRC) Computation

This section describes the algorithm for computing **Standardized Regression Coefficients (SRC)** for sensitivity analysis in hydrologic modeling. It covers both **first-order SRC** (per-parameter sweep) and **total SRC** (multiple regression on all parameters).

---

## 1. Inputs

- **Sweep Data**: For each parameter `p`, a sweep block containing:
  - `x_p`: parameter `p` values across runs.
  - `y_p`: corresponding metric values (e.g., OF, NSE).
- **All-Block Data**:
  - `df_params_all`: matrix of all parameter values (n × k).
  - `df_metric_all`: vector of metric values for the same runs.
- **Parameter Labels**: Names for parameters (e.g., `"CN2 [id-70]"`).

---

## 2. First-Order SRC (Per-Parameter Sweep)

### Goal
Measure the direct sensitivity of the metric to a single parameter, ignoring other parameters.

### Algorithm
1. **Extract sweep data** for parameter `p`:
   - `x ← x_p`
   - `y ← y_p`

2. **Standardize** each vector:
   - \( \tilde{x}_i = \frac{x_i - \bar{x}}{s_x} \)
   - \( \tilde{y}_i = \frac{y_i - \bar{y}}{s_y} \)  
     where \( s_x, s_y \) are the sample standard deviations.

3. **Fit simple linear regression** on standardized data:
   - Model: \( \tilde{y} = \beta^{(1)}_p \tilde{x} + \epsilon \)
   - First-order SRC:
     - \( \mathrm{SRC}^{\text{first}}_p = \beta^{(1)}_p \)
     - This equals the Pearson correlation between `x` and `y`.

4. **Store result**:
   - Record `{PARAM: label_p, Order: "First", Method: "SRC", Sensitivity Index: SRC_first_p}`.

---

## 3. Total SRC (Multiple Regression on All Parameters)

### Goal
Measure each parameter’s contribution when all parameters vary together.

### Algorithm
1. **Prepare input data**:
   - `X ← df_params_all`
   - `y ← df_metric_all`

2. **Standardize**:
   - For each column of `X`:
     - \( \tilde{X}_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j} \)
   - For `y`:
     - \( \tilde{y}_i = \frac{y_i - \mu_y}{\sigma_y} \)

3. **Fit multiple linear regression**:
   - Model:  
     \( \tilde{y} = \alpha + \sum_{j=1}^{k} \beta^{(\text{tot})}_j \tilde{X}_j + \epsilon \)
   - Total SRC for parameter `j`:
     - \( \mathrm{SRC}^{\text{total}}_j = \beta^{(\text{tot})}_j \)

4. **Store result**:
   - Record `{PARAM: label_j, Order: "Total", Method: "SRC", Sensitivity Index: SRC_total_j}`.

---

## 4. Output Structure

The output table should contain:

| PARAM            | Order  | Method | Sensitivity Index |
|------------------|--------|--------|-------------------|
| CN2 [id-70]      | First  | SRC    | 0.45              |
| CN2 [id-70]      | Total  | SRC    | 0.38              |
| ALPHA_BF [id-45] | First  | SRC    | -0.22             |
| ALPHA_BF [id-45] | Total  | SRC    | -0.10             |

---

## 5. Sanity Checks

- Standardization uses **sample** standard deviation (`ddof=1`).
- Remove outliers from `y` before regression if necessary.
- Ensure no NaNs remain after preprocessing.
- Interpret the **sign** of SRC:
  - Positive → increase in parameter increases metric.
  - Negative → increase in parameter decreases metric.

---

## 6. Pseudocode

```text
# FIRST-ORDER SRC
for p in parameters:
    x = sweep_values[p]
    y = metric_values[p]
    xz = standardize(x)
    yz = standardize(y)
    beta_first = OLS_slope(yz ~ xz)   # equals corr(x, y)
    store(PARAM=p_label, Order="First", SRC=beta_first)

# TOTAL SRC
X = params_all_block
y = metric_all_block
Xz = standardize_columns(X)
yz = standardize(y)
beta_totals = OLS_coefficients(yz ~ Xz + intercept)
for j, p in enumerate(parameters):
    store(PARAM=p_label, Order="Total", SRC=beta_totals[j])
````

---

## 7. Notes

* **First-order SRC**: Simple, interpretable, but ignores interactions.
* **Total SRC**: Captures partial effects in a linear model with all parameters included.
* SRC values are **unitless** and comparable across parameters.

```
