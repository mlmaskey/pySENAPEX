'''
Mahesh Lal Maskey, Ph.D. in Hydrologic Sciences
Research Computational Hydrologist
Affiliations: USDA-ARS Sustainable Water Management Research Unit/National Alluvial Aquifer Research Unit
              University of California, Davis & Merced
Copyright@mlmaskey, December 12, 2024
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from configobj import ConfigObj
import seaborn as sns
from Utility.pyAPEXpost import print_progress_bar
from Utility.pyAPEXpost import pyAPEXpost as ap
from Utility.apex_utility import read_sensitive_params
from Utility.apex_utility import split_data
from Utility.easypy import easypy as ep

# from Utility.apex_utility import plot_2_bar  # REMOVED: no V1 plots

warnings.filterwarnings('ignore')
config = ConfigObj('runtime.ini')

PARAM_OFFSET = 70  # used only for display labels like "PARAM [id-70]"


class senanaAPEX:
    def __init__(self, src_dir, config_obj, out_dir, attribute, metric='OF'):
        self.config = config_obj
        self.src_dir = src_dir
        self.file_limits = self.src_dir / 'Utility' / self.config['file_limits']
        self.attribute = attribute
        self.folder = config_obj['dir_sensitivity']

        # ranges + observed
        self.get_range()
        self.df_obs = ap.get_measure(data_dir='Program', file_name='calibration_data.csv')

        # sensitive parameter indices (shift to align with columns used later)
        id_sensitive = read_sensitive_params(self.src_dir)
        for i in range(len(id_sensitive)):
            id_sensitive[i] = id_sensitive[i] + 1

        # bounds for sensitive params
        param_range = self.param_range[:, id_sensitive]
        self.list_bound = list(param_range[:2, :].T)  # [[min,max], ...]

        # files, params, metrics
        self.get_pe_files()
        self.get_params(id_sensitive, is_all=False)
        metric_list = ['RunId', 'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'OF2DC']
        self.read_pem(metric_list)

        # ensure PARAM columns exist before assigning
        if 'PARAM' not in self.params.columns:
            self.params.insert(self.params.shape[1], 'PARAM', 'all')
        if 'PARAM' not in self.pem4criteria.columns:
            self.pem4criteria.insert(self.pem4criteria.shape[1], 'PARAM', 'all')

        df_pem = self.pem4criteria
        df_params = self.params

        self.site = config_obj['Site']
        self.scenario = config_obj['Scenario']

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # thresholds (allow either PBIAS_criteria or PBAIS_criteria)
        cod = float(config_obj.get('COD_criteria', 0.0))
        nse = float(config_obj.get('NSE_criteria', 0.0))
        p_bias = float(config_obj.get('PBIAS_criteria', config_obj.get('PBAIS_criteria', 0.0)))

        self.metric = metric

        # perturbation grid
        maxp = int(config_obj['max_range'])
        deltas = np.linspace(-maxp, maxp, 200)

        n_param = len(self.param_list)
        n_step = len(deltas)
        n_simulate = n_step * (n_param + 1)
        id_vec = np.arange(n_step, n_simulate, n_step)

        # tag param blocks using .loc to avoid chained assignment issues
        for i in range(len(id_vec)):
            start = id_vec[i]
            stop = start + n_step - 1
            df_pem.loc[start:stop, 'PARAM'] = self.param_list[i]
            df_params.loc[start:stop, 'PARAM'] = self.param_list[i]

        # ---- MAIN LOOP per performance metric ----
        for metric in ['OF', 'NSE', 'PBIAS', 'COD']:

            # reset per‑metric scratch frames so things don’t leak/accumulate
            df_sen_summary = pd.DataFrame()
            df_sens = pd.DataFrame()
            df_count = pd.DataFrame()
            df_param_mat = pd.DataFrame()
            df_metric = pd.DataFrame()
            df_src = pd.DataFrame()

            # total SRC using "all" rows
            df_x = df_params[df_params.PARAM == 'all'].drop(['RunId', 'PARAM'], axis=1)
            df_y = df_pem[df_pem.PARAM == 'all'][metric]
            print(f'------Calculating total Standarized Regression Coefficient for {metric}----\n')
            src_total = ap.standarizedRegressionCoefficientTotal(df_x, df_y, intercept=False)

            # counts for "all"
            df_pem_combined = df_pem[df_pem.PARAM == 'all']
            df_pem_criteria_all = df_pem_combined[
                (df_pem_combined.COD > cod) & (df_pem_combined.NSE > nse) & (df_pem_combined.PBIAS < p_bias)
            ]
            df_count1 = pd.DataFrame({
                'COD': (df_pem_combined.COD >= cod).sum(),
                'NSE': (df_pem_combined.NSE >= nse).sum(),
                'PBIAS': (df_pem_combined.PBIAS <= p_bias).sum(),
                'TOTAL': df_pem_criteria_all.shape[0],
                'METRIC': metric
            }, index=['all'])

            # traverse each sensitive parameter’s sweep block
            for j, param in enumerate(self.param_list):
                label = f'PARAM [{id_sensitive[j] - PARAM_OFFSET}]'

                df_pm_j = df_pem[df_pem.PARAM == param]
                df_pa_j = df_params[df_params.PARAM == param]

                df_pem_criteria = df_pm_j[
                    (df_pm_j.COD > cod) & (df_pm_j.NSE > nse) & (df_pm_j.PBIAS < p_bias)
                ]

                # counts per param (correct boolean sums)
                df_count_curr = pd.DataFrame({
                    'COD': (df_pm_j.COD >= cod).sum(),
                    'NSE': (df_pm_j.NSE >= nse).sum(),
                    'PBIAS': (df_pm_j.PBIAS <= p_bias).sum(),
                    'TOTAL': df_pem_criteria.shape[0],
                    'METRIC': metric
                }, index=[label])
                df_count = pd.concat([df_count, df_count_curr], axis=0)

                # build matrices for heatmap
                df_param_mat = pd.concat([df_param_mat, pd.DataFrame(df_pa_j[param].values)], axis=1)
                df_metric = pd.concat([df_metric, pd.DataFrame(df_pm_j[metric].values)], axis=1)

                # local sensitivity table
                df_sen = pd.DataFrame({
                    'PARAM': df_pa_j[param].values,
                    'METRIC': df_pm_j[metric].values
                })
                df_sen['METRIC'] = ep.get_outlier_na(df_sen['METRIC'])
                df_sen.insert(1, 'PARAM_CHANGE', ep.find_change_percent(df_sen.PARAM.values, self.p_best[param]))
                df_sen.insert(3, 'CHANGE_METRIC', ep.find_change_percent(df_sen.METRIC.values, self.pe_best[metric]))
                df_sen.insert(4, 'ABSOLUTE_CHANGE', deltas)
                df_sen.insert(5, 'PARAM_NAME', param)
                df_sen.insert(6, 'NAME', label)

                df_sen_non_nan = df_sen.dropna()

                # correlation + local index
                r2, p_value = ep.corr_test(df_sen_non_nan.PARAM_CHANGE, df_sen_non_nan.CHANGE_METRIC)
                si = ap.sensitivity_index(df_pa_j[param].values, df_pm_j[metric].values)

                df_j = pd.DataFrame({
                    'Nmet': df_pem_criteria.shape[0],
                    'minCOD': df_pem_criteria.COD.min(),
                    'maxCOD': df_pem_criteria.COD.max(),
                    'nCOD': (df_pem_criteria.COD > cod).sum(),
                    'minNSE': df_pem_criteria.NSE.min(),
                    'maxNSE': df_pem_criteria.NSE.max(),
                    'nNSE': (df_pem_criteria.NSE > nse).sum(),
                    'minPBIAS': df_pem_criteria.PBIAS.min(),
                    'maxPBIAS': df_pem_criteria.PBIAS.max(),
                    'nPBIAS': (df_pem_criteria.PBIAS < p_bias).sum(),
                    'MINp': df_pa_j[param].min(),
                    'MAXp': df_pa_j[param].max(),
                    'Rsquared': r2,
                    'p-value': p_value,
                    'SensitivityIndex': si,
                    'PARAM': param
                }, index=[label])
                df_sen_summary = pd.concat([df_sen_summary, df_j], axis=0)
                df_sens = pd.concat([df_sens, df_sen_non_nan], axis=0)

                # individual SRC for this param
                src_first = ap.standarizedRegressionCoefficient(
                    df_pa_j[param].values, df_pm_j[metric].values, intercept=True
                )
                df_src_ = pd.DataFrame({'SRC': src_first, 'Metric': metric}, index=[label])
                df_src = pd.concat([df_src, df_src_], axis=0)

            # finalize per‑metric counts
            df_count = pd.concat([df_count, df_count1], axis=0)
            df_param_mat.columns = self.param_list
            df_param_all = pd.concat([df_params[df_params.PARAM == 'all'][self.param_list], df_param_mat], axis=0)
            df_metric.columns = self.param_list
            df_count.to_csv(f'{out_dir}/Param_within_{metric}.csv')

            # heatmap (percent change vs. best)
            param_labels = [f'PARAM [{id_sensitive[i] - PARAM_OFFSET}]' for i, _ in enumerate(self.param_list)]
            c_bar_title = 'Percent change in objective function value' if metric == 'OF' else f'Percent change in {metric} value'
            df_plot_data = (df_metric - self.pe_best[metric]) * 100 / self.pe_best[metric]
            df_plot_data.columns = param_labels
            df_plot_data.index = [round(item, 2) for item in deltas]
            df_plot_data.to_csv(f'{out_dir}/Heatmap-data_{metric}.csv')

            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            sns.heatmap(df_plot_data, cmap='bwr',
                        vmin=df_plot_data.min().min(),
                        vmax=df_plot_data.max().max(),
                        cbar_kws={'label': c_bar_title}, ax=ax)
            ax.set_ylabel('Percent change in parameters')
            plt.savefig(f'{out_dir}/Heatmap_{metric}.png', dpi=600, bbox_inches="tight")

            # SRC totals aligned to labels
            df_src.insert(1, 'SRC_Total', pd.Series(src_total.values, index=df_src.index), True)

            # ---- SRC PLOT ONLY ----
            df_src_first = pd.DataFrame(
                {'Sensitivity Index': df_src['SRC'].values, 'Order': 'First', 'Method': 'SRC', 'PARAM': param_labels},
                index=param_labels)
            df_src_total = pd.DataFrame(
                {'Sensitivity Index': df_src['SRC_Total'].values, 'Order': 'Total', 'Method': 'SRC', 'PARAM': param_labels},
                index=param_labels)

            df_src_combined = pd.concat([df_src_first, df_src_total], axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(6, 10), tight_layout=True)
            sns.barplot(data=df_src_combined, y='PARAM', x='Sensitivity Index', hue='Order', ci=None, ax=ax)
            ax.grid(True)
            ax.set_ylabel('Parameters')
            ax.set_xscale('log')
            ax.set_title('Standardized Regression Coefficient (SRC)')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
            plt.savefig(f'{out_dir}/Index_{metric}.png', dpi=600, bbox_inches="tight")

            # exports for this metric (SRC‑only now)
            df_param_all.to_csv(f'{out_dir}/Param_{metric}.csv')
            df_src_first.to_csv(f'{out_dir}/Index_first_{metric}.csv')   # SRC first
            df_src_total.to_csv(f'{out_dir}/Index_Total_{metric}.csv')  # SRC total
            df_sen_summary.to_csv(f'{out_dir}/Stat_Summary_{metric}.csv')

        # ---- bulk outputs (unchanged) ----
        print('------- importing results ------\n')
        df_wyld = self.import_output(location='outlet', variable='WYLD'); df_wyld.to_csv(f'{out_dir}/Daily_WYLD.csv')
        df_rus2 = self.import_output(location='outlet', variable='RUS2'); df_rus2.to_csv(f'{out_dir}/Daily_RUS2.csv')
        df_ysd  = self.import_output(location='outlet', variable='YSD');  df_ysd.to_csv(f'{out_dir}/Daily_YSD.csv')
        df_lai  = self.import_output(location='basin',  variable='LAI');  df_lai.to_csv(f'{out_dir}/Daily_LAI.csv')
        df_biom = self.import_output(location='basin',  variable='BIOM'); df_biom.to_csv(f'{out_dir}/Daily_BIOM.csv')
        df_prcp = self.import_output(location='basin',  variable='PRCP'); df_prcp.to_csv(f'{out_dir}/Daily_PRCP.csv')
        df_stl  = self.import_output(location='basin',  variable='STL');  df_stl.to_csv(f'{out_dir}/Daily_STL.csv')
        df_std  = self.import_output(location='basin',  variable='STD');  df_std.to_csv(f'{out_dir}/Daily_STD.csv')
        df_stdl = self.import_output(location='basin',  variable='STDL'); df_stdl.to_csv(f'{out_dir}/Daily_STDL.csv')
        df_et   = self.import_output(location='basin',  variable='ET');   df_et.to_csv(f'{out_dir}/Daily_ET.csv')
        df_pet  = self.import_output(location='basin',  variable='PET');  df_pet.to_csv(f'{out_dir}/Daily_PET.csv')
        df_dprk = self.import_output(location='basin',  variable='DPRK'); df_dprk.to_csv(f'{out_dir}/Daily_DPRK.csv')
        df_tp   = self.import_output(location='basin',  variable='TP');   df_tp.to_csv(f'{out_dir}/Daily_TP.csv')
        df_tn   = self.import_output(location='basin',  variable='TN');   df_tn.to_csv(f'{out_dir}/Daily_TN.csv')
        df_yldg = self.import_output(location='annual', variable='YLDG'); df_yldg.to_csv(f'{out_dir}/Annual_YLDG.csv')
        df_yldf = self.import_output(location='annual', variable='YLDF'); df_yldf.to_csv(f'{out_dir}/Annual_YLDF.csv')
        df_biom = self.import_output(location='annual', variable='BIOM'); df_biom.to_csv(f'{out_dir}/Annual_BIOM.csv')
        df_ws   = self.import_output(location='annual', variable='WS');   df_ws.to_csv(f'{out_dir}/Annual_WS.csv')
        df_ns   = self.import_output(location='annual', variable='NS');   df_ns.to_csv(f'{out_dir}/Annual_NS.csv')
        df_ps   = self.import_output(location='annual', variable='PS');   df_ps.to_csv(f'{out_dir}/Annual_PS.csv')
        df_ts   = self.import_output(location='annual', variable='TS');   df_ts.to_csv(f'{out_dir}/Annual_TS.csv')

    def get_range(self):
        df_param_limit = pd.read_csv(self.file_limits, index_col=0, encoding="ISO-8859-1")
        arr = df_param_limit.iloc[1:, :].to_numpy().astype(float)
        self.param_range = np.squeeze(np.asarray(arr))
        return self

    def get_pe_files(self):
        if self.attribute == 'WYLD':
            file_pe = os.path.join(self.folder, 'Statistics_runoff.csv')
        elif self.attribute == 'YSD':
            file_pe = os.path.join(self.folder, 'Statistics_sediment.csv')
        else:
            file_pe = os.path.join(self.folder, f'Statistics_{self.attribute}.csv')
        file_parameter = os.path.join(self.folder, 'APEXPARM.csv')
        self.file_pem = file_pe
        self.file_param = file_parameter
        return self

    def get_params(self, id_sensitive, is_all):
        df_params = pd.read_csv(self.file_param)
        df_params.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        self.params_all = df_params

        if is_all:
            self.params = df_params.copy()
            self.param_list = [c for c in df_params.columns if c != 'RunId']
        else:
            cols = df_params.columns
            self.param_list = cols[id_sensitive].tolist()
            df_param = df_params[self.param_list].copy()
            df_param.insert(0, 'RunId', df_params['RunId'].values)
            self.params = df_param

        self.p_best = self.params.iloc[0, :]
        self.params = self.params.iloc[1, :]
        return self

    def read_pem(self, metric_list):
        df_pem = ap.get_stats(self.file_pem)
        df_pem_obs = df_pem[metric_list].copy()
        df_pem_obs.columns = ['RunId', 'COD', 'RMSE', 'NRMSE', 'NSE', 'PBIAS', 'OF']
        pe_best = df_pem_obs.iloc[0, :].copy()
        df_pem_obs = df_pem_obs.iloc[1, :]
        self.pem = df_pem
        self.pem4criteria = df_pem_obs
        self.pe_best = pe_best
        return self

    def assign_dates(self, df_mod, n_warm, n_calib_year):
        start_year = df_mod.Year.values[0]
        cal_start = start_year + n_warm
        cal_end = cal_start + n_calib_year
        val_end = self.df_obs.Year.iloc[-1]
        val_end_date = self.df_obs.Date.iloc[-1]
        return start_year, cal_start, cal_end, val_end, val_end_date

    def import_output(self, location, variable):
        print(f'\nExporting {variable} data')
        stage_vec = []
        n_runs = self.pem4criteria.shape[0]
        n_warm = int(self.config['warm_years'])
        n_calib_year = int(self.config['calib_years'])
        df_out = pd.DataFrame()
        print_progress_bar(0, n_runs + 1, prefix='Progress', suffix='Complete', length=50, fill='█')
        for i in range(1, n_runs + 1):
            if location == 'outlet':
                file_outlet = os.path.join(self.folder, f'daily_outlet_{i:07}.csv.csv')
                data = pd.read_csv(file_outlet)
                data.index = pd.to_datetime(data.Date)
                data = data.drop(['Date', 'Y', 'M', 'D'], axis=1)
                data.insert(0, 'Year', data.index.year, True)
                try:
                    start_year, cal_start, cal_end, val_end, val_end_date = self.assign_dates(data, n_warm, n_calib_year)
                except Exception:
                    continue
                df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                df_data_val = df_data_val[df_data_val.index <= val_end_date]
            elif location == 'basin':
                file_basin = os.path.join(self.folder, f'daily_basin_{i:07}.csv.csv')
                data = pd.read_csv(file_basin)
                data.index = pd.to_datetime(data.Date)
                data = data.drop(['Date', 'Y', 'M', 'D'], axis=1)
                data.insert(0, 'Year', data.index.year, True)
                try:
                    start_year, cal_start, cal_end, val_end, val_end_date = self.assign_dates(data, n_warm, n_calib_year)
                except Exception:
                    continue
                df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                df_data_val = df_data_val[df_data_val.index <= val_end_date]
            else:
                file_annual = os.path.join(self.folder, f'annual_{i:07}.csv.csv')
                data = pd.read_csv(file_annual)
                data.index = data.YR
                data = data.drop(['Unnamed: 0', 'YR'], axis=1)
                data.insert(0, 'Year', data.index, True)
                try:
                    start_year, cal_start, cal_end, val_end, val_end_date = self.assign_dates(data, n_warm, n_calib_year)
                except Exception:
                    continue
                df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                df_data_val = df_data_val[df_data_val.index <= val_end]

            df_data_cal.insert(df_data_cal.shape[1], 'Stage', 'Calibration', True)
            df_data_val.insert(df_data_val.shape[1], 'Stage', 'Validation', True)
            df_data = pd.concat([df_data_cal, df_data_val], axis=0)
            stage_vec = df_data.Stage.values
            df_series = pd.DataFrame(df_data[variable]).copy()
            df_series.columns = [f'{i}']
            df_out = pd.concat([df_out, df_series], axis=1)
            print_progress_bar(i, n_runs + 1, prefix=f'Progress: {i}', suffix='Complete', length=50, fill='█')
        df_out.insert(0, 'Stage', stage_vec, True)
        return df_out
