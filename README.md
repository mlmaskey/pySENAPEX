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
