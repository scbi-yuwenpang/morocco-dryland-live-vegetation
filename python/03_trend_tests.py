"""Reference implementation of yearly and monthly trend statistics.

Inputs: station-level RF predictions with columns
    Sensor, Habitat, Station, date, F_alive_Pred
(daily/acquisition-level records, possibly multiple records per station-month).

Procedure: mean within Station × month; equal-weight stations within
Habitat × month; annual means use >=9 valid months; test 2017–2025.
Each satellite's seven habitat tests form a separate BH-FDR family. The monthly
Hamed–Rao test acts on calendar-month-deseasonalized monthly anomalies.

"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.stats import theilslopes
from statsmodels.stats.multitest import multipletests


def _fdr_within_sensor(tab, p_col, q_col):
    tab[q_col] = np.nan
    for sensor, idx in tab.groupby('Sensor').groups.items():
        sub = tab.loc[idx, p_col]
        keep = sub.notna()
        if keep.any():
            tab.loc[sub.index[keep], q_col] = multipletests(sub[keep], method='fdr_bh')[1]
    return tab


def analyze(df, start_year=2017, end_year=2025, min_months=9):
    required = {'Sensor','Habitat','Station','date','F_alive_Pred'}
    if missing := required-set(df.columns):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d['F_alive_Pred'] = pd.to_numeric(d['F_alive_Pred'], errors='coerce')
    d = d.dropna(subset=['date','F_alive_Pred','Station','Habitat','Sensor'])
    d = d[d.date.dt.year.between(start_year,end_year)].copy()
    d['Year'] = d.date.dt.year
    d['Month'] = d.date.dt.month

    # Equal-weight independent stations, not equal-weight all image acquisitions.
    station_month = d.groupby(['Sensor','Habitat','Station','Year','Month'], as_index=False).agg(
        predicted=('F_alive_Pred', 'mean'))
    monthly = station_month.groupby(['Sensor','Habitat','Year','Month'], as_index=False).agg(
        F_alive_Pred=('predicted','mean'), n_stations=('Station','nunique'))
    monthly['date'] = pd.to_datetime(dict(year=monthly.Year, month=monthly.Month, day=1))
    annual = monthly.groupby(['Sensor','Habitat','Year'], as_index=False).agg(
        annual_mean=('F_alive_Pred','mean'), n_months=('Month','nunique'))
    annual = annual[annual.n_months >= min_months].copy()

    a_rows = []
    for (sensor, habitat), sub in annual.groupby(['Sensor','Habitat']):
        sub = sub.sort_values('Year')
        if len(sub) < 3:
            continue
        y, x = sub.annual_mean.to_numpy(), sub.Year.to_numpy()
        sen = theilslopes(y, x, alpha=0.95)
        test = mk.original_test(y)
        a_rows.append(dict(Sensor=sensor, Habitat=habitat, n_years=len(sub),
            start_year=int(x.min()), end_year=int(x.max()),
            Sen_slope_per_year=sen.slope, Sen_CI_low=sen.low_slope,
            Sen_CI_high=sen.high_slope, MK_tau=test.Tau, MK_p=test.p,
            MK_direction=test.trend))
    annual_tests = pd.DataFrame(a_rows)
    if not annual_tests.empty:
        annual_tests = _fdr_within_sensor(annual_tests,'MK_p','FDR_q')

    # Calendar-month climatology computed separately by sensor and habitat.
    monthly['climatology'] = monthly.groupby(['Sensor','Habitat','Month'])['F_alive_Pred'].transform('mean')
    monthly['anomaly'] = monthly['F_alive_Pred']-monthly['climatology']
    m_rows = []
    for (sensor, habitat), sub in monthly.groupby(['Sensor','Habitat']):
        sub = sub.sort_values('date')
        y = sub.anomaly.to_numpy()
        if len(y) < 12:
            continue
        # Hamed–Rao assumes time-ordered, evenly spaced samples. If monthly
        # gaps exist, verify they are acceptable before interpreting the test.
        complete_months = pd.period_range(sub.date.min(), sub.date.max(), freq='M')
        if len(complete_months) != len(sub):
            # Do not silently compress missing months into a regular series.
            print(f'WARNING: {sensor} / {habitat} has missing months; review Hamed–Rao assumptions.')
        test = mk.hamed_rao_modification_test(y)
        m_rows.append(dict(Sensor=sensor, Habitat=habitat, n_months=len(y),
            HR_MK_tau=test.Tau, HR_MK_p=test.p, HR_MK_direction=test.trend))
    monthly_tests = pd.DataFrame(m_rows)
    if not monthly_tests.empty:
        monthly_tests = _fdr_within_sensor(monthly_tests,'HR_MK_p','HR_FDR_q')
    return station_month, monthly, annual, annual_tests, monthly_tests


def main(input_path, output_dir):
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    results = analyze(pd.read_csv(input_path))
    for name, table in zip(['station_month','habitat_month','habitat_year',
                             'annual_tests','monthly_hamed_rao'],results):
        table.to_csv(out/f'{name}_REFERENCE.csv',index=False)
        print(name,table.shape)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True)
    args=p.parse_args()
    main(args.input,args.output_dir)
