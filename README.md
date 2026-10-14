# Budhi Gandaki: trends and anomalies only

This repository includes the original daily CSV, reproducible Python code,
current analysis tables, all variable plots and an HTML report. See
[FINDINGS.md](FINDINGS.md) for the interpretation and follow-up investigations.
Download or clone the repository and open `trend_anomaly_output/report.html`
in a browser to view the report with its figures.

```powershell
python -m pip install -r requirements.txt
python budhi_analysis.py
```

Open `trend_anomaly_output/report.html`. All varying numeric variables are
analyzed; identifiers, date fields and constant variables are excluded.
The input CSV is preserved. Every analyzed variable has a plot by default,
with units on its axes and Sen slope. Monthly anomalies are omitted.

## Trend analysis

- Annual original Mann–Kendall: S, tie-corrected variance, continuity-corrected
  Z, two-sided p-value, Kendall tau and trend direction.
- Annual Sen slope per year and decade with 95% confidence bounds.
- Separate DJF/MAM/JJA/SON Mann–Kendall and Sen slopes. December is assigned
  to the following DJF year.
- Seasonal Mann–Kendall compares each calendar month across years; seasonal
  Sen slope uses the median of all within-month pairwise slopes. This test
  uses monthly mean daily values, including daily flux rates, to avoid month
  length effects. It does not estimate a seasonal Sen confidence interval.
- Benjamini–Hochberg adjusted p-values and significance flags within each table.
- Annual lag-1 correlation helps assess independence. **No detrending or
  prewhitening is applied.** Original MK p-values and Sen confidence intervals
  assume independent observations; serial dependence can overstate significance.
  Seasonal MK sums within-month variances without cross-month covariance correction.

## Anomaly analysis

The default requested reference is **1945–1975**. The CSV starts in 1955, so
calculations use the available **1955–1975** records; 1945–1954 is not reconstructed.
Anomaly = value minus baseline mean;
standardized anomaly = anomaly divided by baseline sample standard deviation.
Zero baseline SD gives missing standardized anomalies.

- Daily: fixed calendar month/day climatology (leap day separately).
- Annual: baseline complete-year mean.
- DJF/MAM/JJA/SON: corresponding baseline season-year mean.

CSV tables include all variables, baseline means/SD/sample counts, anomalies,
standardized anomalies, period coverage and trend statistics. Plots show annual
values with Sen lines and annual anomalies. If no complete annual records
exist, a plot shows available daily values and daily anomalies and explicitly
marks the annual trend unavailable.
No forecasting, correlation analysis, decomposition, spectra, change-point
screening or return-period analysis is included.

Two additional plots show annual rainfall and snowfall percentages of total
precipitation, each with a Mann–Kendall test, Sen slope and annual anomalies.
Percentage = 100 × annual component total / annual precipitation total.
Snowfall uses water-equivalent mm. Component totals and precipitation must
have the same fully observed days. Zero precipitation and incomplete years
give missing percentages. Percentage anomalies and slopes are in percentage
points; ratios are not clipped or forced to sum to 100% if the source
components differ slightly from total precipitation.

Units follow source column names and the
[ECMWF ERA5 parameter documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=669811810).
They are also listed in `variable_aggregation_rules.csv`. Daily flux totals
use mm/year, temperatures use °C, soil moisture uses m³/m³, and fractions and
indicators have explicit dimensionless labels. Original source accumulation
columns retain their daily accumulation units when averaged across a year.

## Options

```powershell
python budhi_analysis.py --plot-all
python budhi_analysis.py --variables precipitation_mm_day runoff_mm_day temperature_2m_C_mean
python budhi_analysis.py --baseline-start 1981-01-01 --baseline-end 2010-12-31 --output alternative_baseline
```

Only fully observed calendar months, years and seasons enter their respective
aggregated results. Known daily water flux columns listed in `FLUX` are summed;
other variables are averaged. Consequently slopes for annual flux totals have
units of (mm/year)/year, while state-variable slopes have source units/year.
Existing rolling and derived columns are included if varying, but their period
means are descriptive; verify their original formulas and units before interpreting.

Tiny negative water flux values between -0.001 and 0 mm are clipped to zero;
more negative values become missing. Temperatures and signed water balances
retain negative values. Counts are recorded in `cleaning_log.csv`.

The reported flood is **October 11, 2026**, but data ends **October 5, 2026**.
Daily pre-event anomalies are available; October 2026, annual 2026 and SON
2026 aggregate results are missing because those periods are incomplete.
No data is invented for October 6–11. ERA5 point runoff is not measured river
discharge, and ERA5 snow depth is water-equivalent depth. Verify extraction
metadata before making physical or flood attribution claims.

References: [Mann (1945)](https://doi.org/10.2307/1907187),
[Sen (1968)](https://doi.org/10.1080/01621459.1968.10480934),
[Hirsch et al. (1982)](https://doi.org/10.1029/WR018i001p00107).
