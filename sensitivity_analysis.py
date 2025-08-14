# run_sa.py
# -*- coding: utf-8 -*-
"""
Run post-analysis sensitivity analysis (PBSA + SRC) for APEX/APEXgraze.

Created on Wed Dec 21 07:42:35 2022
@author: Mahesh.Maskey
"""

import os
from pathlib import Path
from configobj import ConfigObj
from pySAAPEX import senanaAPEX

# Determine source directory
src_dir = Path(os.path.dirname(os.path.realpath(__file__)))

# Load configuration
config = ConfigObj(str(src_dir / 'runtime.ini'))

# Set site and scenario
site = 'Farm_1'
scenario = 'grazing'
config['Site'] = site
config['Scenario'] = scenario

# Output directory
out_dir = src_dir / '../../../post_analysis/sensitivity_analysis' / site / scenario
out_dir.mkdir(parents=True, exist_ok=True)

# Run sensitivity analysis
sen_obj = senanaAPEX(src_dir, config, str(out_dir),
                     attribute='runoff', metric='OF')

print(f"Sensitivity analysis complete. Results in: {out_dir}")
