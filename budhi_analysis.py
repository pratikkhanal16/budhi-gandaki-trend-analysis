"""Only Mann-Kendall/Sen trend analysis and baseline anomalies for daily ERA5."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

META = {'system:index','day','month','year','doy','decade','hydro_year',
        'season_year','latitude','longitude','hourly_records'}
FLUX = {'precipitation_mm_day','runoff_mm_day','snowmelt_mm_day_water_equivalent',
        'snowfall_mm_day_water_equivalent','rainfall_mm_day','surface_runoff_mm_day',
        'subsurface_runoff_mm_day','convective_precipitation_mm_day',
        'large_scale_precipitation_mm_day','PET_mm_day','climatic_water_balance_mm',
        'climatic_water_balance','melt_plus_rain_mm','rain_on_snow_mm'}
CORE = ['precipitation_mm_day','runoff_mm_day','snowmelt_mm_day_water_equivalent',
        'temperature_2m_C_mean','volumetric_soil_water_layer_1_mean','snow_depth_mean',
        'soil_temperature_level_1_C_mean','zero_degree_level_mean','PET_mm_day']


def source_unit(c):
    """Units of stored daily values; derived units follow named source variables."""
    aliases = {'T':'temperature_2m_C_mean','P':'precipitation_mm_day',
               'SM1':'volumetric_soil_water_layer_1_mean','Q':'runoff_mm_day',
               'SD':'snow_depth_mean','MELT':'snowmelt_mm_day_water_equivalent',
               'ZERO':'zero_degree_level_mean',**{f'ST{i}':f'soil_temperature_level_{i}_C_mean' for i in range(1,5)}}
    for suffix in ['_clim','_anomaly']:
        if c.endswith(suffix) and c[:-len(suffix)] in aliases:
            return source_unit(aliases[c[:-len(suffix)]])
    short = {'runoff_clim':'mm/day','snow_depth_clim':'m water equivalent',
             'snowmelt_clim':'mm/day','zero_level_clim':'m',
             'precipitation_anomaly':'mm/day','runoff_anomaly':'mm/day',
             'snow_depth_anomaly':'m water equivalent','snowmelt_anomaly':'mm/day',
             'zero_level_anomaly':'m','climatic_water_balance':'mm/day',
             'climatic_water_balance_mm':'mm/day','rain_on_snow_mm':'mm/day',
             'melt_plus_rain_mm':'mm/day','thawed_layer_count':'layers (0–4)',
             'k_index_mean':'K','total_totals_index_mean':'K'}
    if c in short:
        return short[c]
    if c.startswith('AN_') and c.endswith('_z'):
        return 'dimensionless (z-score)'
    if re.fullmatch(r'P\d+|runoff_\d+|melt_\d+',c):
        return 'mm (rolling total)'
    if c.startswith('SM1_') or 'volumetric_soil_water' in c:
        return 'm³/m³'
    if '_TDD' in c or '_FDD' in c:
        return '°C·day'
    if '_C_' in c or '_air_offset' in c or c.endswith('_gradient'):
        return '°C'
    if '_mm_day' in c:
        return 'mm/day'
    if '_MJ_m2_day' in c:
        return 'MJ/m²/day'
    if 'relative_humidity_percent' in c:
        return '%'
    if 'convective_available_potential_energy' in c:
        return 'J/kg'
    if 'mean_vertically_integrated_moisture_divergence' in c:
        return 'kg/m²/s'
    if 'water_vapour_flux' in c:
        return 'kg/m/s'
    if c.startswith('total_column_'):
        return 'kg/m²'
    if 'pressure' in c:
        return 'Pa'
    if 'wind' in c:
        return 'm/s'
    if 'snow_density' in c:
        return 'kg/m³'
    if 'snow_depth' in c or c=='skin_reservoir_content_mean':
        return 'm water equivalent'
    if c in {'boundary_layer_height_mean','cloud_base_height_mean','zero_degree_level_mean'}:
        return 'm'
    if 'leaf_area_index' in c:
        return 'm²/m²'
    if c.endswith('_sum'):
        return 'J/m² (daily accumulation)' if 'radiation' in c or 'heat_flux' in c else 'm water equivalent/day'
    if any(k in c for k in ['_cover','albedo','fraction']):
        return 'dimensionless (0–1)'
    if (c in {'snow_present','substantial_snow','positive_temp_day','negative_temp_day'}
            or c.endswith('_day') or 'transition' in c):
        return 'indicator (0/1)'
    raise ValueError(f'Unit mapping needed for {c}')


def annual_unit(c):
    if c in FLUX:
        return 'mm/year'
    unit = source_unit(c)
    return 'fraction of days (0–1)' if unit=='indicator (0/1)' else unit


def precipitation_shares(d):
    """Ratios of annual component totals to annual precipitation, not mean daily ratios."""
    names = ['precipitation_mm_day','rainfall_mm_day','snowfall_mm_day_water_equivalent']
    if not set(names)<=set(d):
        return pd.DataFrame()
    # Use the same complete days for numerator and denominator.
    common = d[names].where(d[names].notna().all(axis=1),np.nan)
    yearly,_ = aggregate(common,'YS')
    denominator = yearly.precipitation_mm_day.where(yearly.precipitation_mm_day>0)
    return pd.DataFrame({'rainfall_percent_of_ppt':100*yearly.rainfall_mm_day/denominator,
                         'snowfall_percent_of_ppt':100*yearly.snowfall_mm_day_water_equivalent/denominator})


def mann_kendall(values):
    """Original two-sided MK, tie-corrected variance and continuity-corrected Z.

    Null: no monotonic trend. Assumes independent observations. No detrending.
    """
    v = np.asarray(values,dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return dict(n=n,S=np.nan,var_S=np.nan,Z=np.nan,p_value=np.nan,tau=np.nan)
    S = float(sum(np.sign(v[i+1:]-v[i]).sum() for i in range(n-1)))
    _,counts = np.unique(v,return_counts=True)
    var = float((n*(n-1)*(2*n+5)-np.sum(counts*(counts-1)*(2*counts+5)))/18)
    Z = (S-np.sign(S))/np.sqrt(var) if var > 0 else 0.
    return dict(n=n,S=S,var_S=var,Z=Z,p_value=float(2*stats.norm.sf(abs(Z))),
                tau=S/(n*(n-1)/2))


def fdr(p):
    """Benjamini-Hochberg adjusted p-values, preserving missing values."""
    a = np.asarray(p,dtype=float)
    good = np.flatnonzero(np.isfinite(a))
    result = np.full(len(a),np.nan)
    order = good[np.argsort(a[good])]
    if len(order):
        adjusted = a[order]*len(order)/np.arange(1,len(order)+1)
        result[order] = np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def write(df,name,out):
    df.to_csv(out/'tables'/f'{name}.csv',index=True)


def aggregate(d,rule,coverage_required=1.):
    g = d.resample(rule)
    values = g.mean()
    flux = [c for c in d if c in FLUX]
    values[flux] = g[flux].sum(min_count=1)
    expected = pd.Series({t:t.days_in_month if rule=='MS' else
                          366 if t.is_leap_year else 365 for t in values.index})
    coverage = g.count().div(expected,axis=0)
    return values.where(coverage>=coverage_required),coverage


def trends(data,alpha):
    rows = []
    for c in data:
        s = data[c].dropna()
        if len(s)<10:
            continue
        x = s.index.year.to_numpy(dtype=float)
        mk = mann_kendall(s)
        sen = stats.theilslopes(s.to_numpy(),x,alpha=.95)
        lag_pairs = pd.concat([data[c],data[c].shift(1)],axis=1).dropna()
        lag1 = (lag_pairs.iloc[:,0].corr(lag_pairs.iloc[:,1])
                if len(lag_pairs)>=3 and (lag_pairs.nunique()>1).all() else np.nan)
        rows.append(dict(variable=c,**mk,first_year=int(x[0]),last_year=int(x[-1]),
                         sen_slope_per_year=sen.slope,sen_slope_per_decade=sen.slope*10,
                         sen_ci95_low_per_decade=sen.low_slope*10,
                         sen_ci95_high_per_decade=sen.high_slope*10,
                         sen_intercept=sen.intercept,
                         lag1_annual_correlation=lag1,
                         trend_unadjusted='increasing' if mk['p_value']<alpha and mk['S']>0 else
                         'decreasing' if mk['p_value']<alpha and mk['S']<0 else 'not significant'))
    result = pd.DataFrame(rows).set_index('variable')
    result['p_fdr_bh'] = fdr(result.p_value)
    result['significant_fdr'] = result.p_fdr_bh < alpha
    return result


def seasonal_mk(monthly_means,alpha):
    """Compare each calendar month across years; combine monthwise MK scores.

    Variance is the sum of month variances (no cross-month covariance correction).
    Seasonal Sen slope is the median of same-month pair slopes in actual years.
    """
    rows = []
    for c in monthly_means:
        S = variance = pairs = n = 0
        slopes = []
        for month in range(1,13):
            s = monthly_means.loc[monthly_means.index.month==month,c].dropna()
            if len(s)<3:
                continue
            mk = mann_kendall(s)
            S += mk['S']; variance += mk['var_S']; n += len(s)
            pairs += len(s)*(len(s)-1)/2
            x,y = s.index.year.to_numpy(),s.to_numpy()
            for i in range(len(s)-1):
                slopes.extend(((y[i+1:]-y[i])/(x[i+1:]-x[i])).tolist())
        if not slopes:
            continue
        z = (S-np.sign(S))/np.sqrt(variance) if variance>0 else 0.
        p = float(2*stats.norm.sf(abs(z)))
        rows.append(dict(variable=c,n_months=n,S=S,var_S=variance,Z=z,p_value=p,
                         tau=S/pairs,seasonal_sen_per_year=np.median(slopes),
                         seasonal_sen_per_decade=np.median(slopes)*10,
                         trend_unadjusted='increasing' if p<alpha and S>0 else
                         'decreasing' if p<alpha and S<0 else 'not significant'))
    result = pd.DataFrame(rows).set_index('variable')
    result['p_fdr_bh'] = fdr(result.p_value)
    result['significant_fdr'] = result.p_fdr_bh<alpha
    return result


def calendar_seasons(d):
    month = d.index.month
    labels = np.select([np.isin(month,[12,1,2]),np.isin(month,[3,4,5]),
                        np.isin(month,[6,7,8])],['DJF','MAM','JJA'],default='SON')
    years = d.index.year+(month==12).astype(int)
    result = {}
    for season in ['DJF','MAM','JJA','SON']:
        mask = labels==season
        subset = d.loc[mask]
        group = subset.groupby(years[mask])
        val = group.mean()
        flux = [c for c in d if c in FLUX]
        val[flux] = group[flux].sum(min_count=1)
        expected = pd.Series({year:sum(pd.Timestamp(year-1 if m==12 else year,m,1).days_in_month
                                     for m in {'DJF':[12,1,2],'MAM':[3,4,5],
                                               'JJA':[6,7,8],'SON':[9,10,11]}[season])
                              for year in val.index})
        val = val.where(group.count().div(expected,axis=0)>=1.)
        val.index = pd.to_datetime(val.index.astype(str)+'-01-01')
        result[season] = val
    return result


def anomalies(data,baseline_start,baseline_end,group_keys=None):
    ref = data.loc[baseline_start:baseline_end]
    if group_keys is None:
        mean,sd,n = ref.mean(),ref.std(),ref.count()
        an = data-mean
        z = an/sd.replace(0,np.nan)
        climatology = pd.DataFrame({'baseline_mean':mean,'baseline_sd':sd,'baseline_n':n})
    else:
        ref_keys = group_keys[(data.index>=pd.Timestamp(baseline_start)) &
                              (data.index<=pd.Timestamp(baseline_end))]
        grouped = ref.groupby(ref_keys)
        mean,sd,n = grouped.mean(),grouped.std(),grouped.count()
        mapped_mean = mean.reindex(group_keys).set_axis(data.index)
        mapped_sd = sd.reindex(group_keys).set_axis(data.index).replace(0,np.nan)
        an = data-mapped_mean
        z = an/mapped_sd
        climatology = pd.concat({'mean':mean,'sd':sd,'n':n},axis=1)
    return an,z,climatology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('Budhi_daily_advanced_python.csv'))
    parser.add_argument('--output',type=Path,default=Path('trend_anomaly_output'))
    parser.add_argument('--baseline-start',default='1945-01-01')
    parser.add_argument('--baseline-end',default='1975-12-31')
    parser.add_argument('--alpha',type=float,default=.05)
    parser.add_argument('--variables',nargs='+',help='Optional variable names; default all varying numeric data columns')
    parser.add_argument('--plot-all',action='store_true',help='Plot every analyzed variable (also the default)')
    args = parser.parse_args()
    if not 0<args.alpha<1:
        parser.error('alpha must be between 0 and 1')
    out = args.output
    for sub in ['tables','figures']:
        (out/sub).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'analysis.log',mode='w')])
    raw = pd.read_csv(args.input)
    raw['date'] = pd.to_datetime(raw.date,errors='raise')
    if raw.date.isna().any() or raw.date.duplicated().any():
        raise ValueError('Invalid or duplicate dates must be resolved first.')
    if len(raw[['latitude','longitude']].drop_duplicates())!=1:
        raise ValueError('Expected one location.')
    raw = raw.sort_values('date').set_index('date')
    if not (raw.index==raw.index.normalize()).all():
        raise ValueError('Expected daily dates without time components.')
    index = pd.date_range(raw.index.min(),raw.index.max(),freq='D',name='date')
    d = raw.select_dtypes('number').drop(columns=list(META),errors='ignore').reindex(index)
    d = d.replace([np.inf,-np.inf],np.nan)
    quality = pd.DataFrame({'missing_n':d.isna().sum(),'unique_n':d.nunique(),
                            'min':d.min(),'max':d.max(),'negative_n':d.lt(0).sum()})
    write(quality,'data_quality',out)
    if args.variables:
        unknown = set(args.variables)-set(d)
        if unknown:
            raise ValueError(f'Unknown numeric variables: {sorted(unknown)}')
        d = d[args.variables]
    d = d.loc[:,d.nunique()>1]
    if d.empty:
        raise ValueError('No varying numeric variables selected.')
    corrections = []
    for c in (FLUX-{'climatic_water_balance','climatic_water_balance_mm'}) & set(d):
        tiny = d[c].between(-.001,0,inclusive='left'); large = d[c]<-.001
        corrections.append(dict(variable=c,tiny_negatives_clipped=int(tiny.sum()),
                                larger_negatives_set_missing=int(large.sum())))
        d.loc[tiny,c] = 0.; d.loc[large,c] = np.nan
    write(pd.DataFrame(corrections),'cleaning_log',out)
    ref = d.loc[args.baseline_start:args.baseline_end]
    if len(ref)<3650:
        raise ValueError('Choose a baseline with at least 10 years available.')
    metadata = pd.DataFrame({'annual_monthly_statistic':['total' if c in FLUX else 'mean' for c in d],
                             'daily_source_unit':[source_unit(c) for c in d],
                             'annual_plot_unit':[annual_unit(c) for c in d],
                             'annual_sen_slope_unit':[annual_unit(c)+'/decade' for c in d]},index=d.columns)
    write(metadata,'variable_aggregation_rules',out)
    logging.info('Analyzing %s variables: original MK and Sen slopes',len(d.columns))
    monthly,mcov = aggregate(d,'MS')
    annual,acov = aggregate(d,'YS')
    write(monthly,'monthly_values',out); write(annual,'annual_values',out)
    write(mcov,'monthly_coverage',out); write(acov,'annual_coverage',out)
    annual_trends = trends(annual,args.alpha)
    write(annual_trends,'annual_mann_kendall_sen',out)
    excluded = pd.DataFrame({'complete_years':annual.count()})
    excluded = excluded.loc[excluded.complete_years<10]
    write(excluded,'annual_trends_insufficient_data',out)
    # Monthly mean daily flux, avoiding month length effects; state means unchanged.
    monthly_means = d.resample('MS').mean().where(mcov>=1.)
    write(seasonal_mk(monthly_means,args.alpha),'seasonal_mann_kendall_sen',out)
    for season,val in calendar_seasons(d).items():
        write(val,f'{season}_values',out)
        write(trends(val,args.alpha),f'{season}_mann_kendall_sen',out)
        an,z,clim = anomalies(val,args.baseline_start,args.baseline_end)
        write(an,f'{season}_anomalies',out); write(z,f'{season}_standardized_anomalies',out)
        write(clim,f'{season}_baseline_climatology',out)
    logging.info('Computing daily and annual anomalies (no detrending)')
    daily_an,daily_z,daily_clim = anomalies(d,args.baseline_start,args.baseline_end,index.strftime('%m-%d'))
    year_an,year_z,year_clim = anomalies(annual,args.baseline_start,args.baseline_end)
    for label,an,z,clim in [('daily',daily_an,daily_z,daily_clim),('annual',year_an,year_z,year_clim)]:
        write(an,f'{label}_anomalies',out); write(z,f'{label}_standardized_anomalies',out)
        write(clim,f'{label}_baseline_climatology',out)
    # No trend removal: anomaly = value minus fixed baseline climatology.
    plots = list(d)
    for c in plots:
        logging.info('Plot: %s',c)
        s = annual[c].dropna()
        fig,axes = plt.subplots(2,1,figsize=(12,7),sharex=True)
        axes[0].plot(s.index.year,s,lw=1,label='Annual '+('total' if c in FLUX else 'mean'))
        if c in annual_trends.index:
            row = annual_trends.loc[c]
            line = row.sen_slope_per_year*s.index.year+row.sen_intercept
            axes[0].plot(s.index.year,line,color='crimson',label='Sen trend line')
            axes[0].set_title(f'MK p={row.p_value:.4g}; FDR p={row.p_fdr_bh:.4g}; '
                              f'Sen slope={row.sen_slope_per_decade:.4g} {annual_unit(c)}/decade')
        elif s.empty:
            axes[0].plot(d.index,d[c],lw=.5,label='Available daily observations')
            axes[0].set_title('No complete calendar years: annual trend unavailable')
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel(annual_unit(c) if not s.empty else source_unit(c))
        if s.empty:
            axes[1].plot(daily_an.index,daily_an[c],lw=.5)
            axes[1].set_ylabel(f'Daily anomaly ({source_unit(c)})')
        else:
            axes[1].bar(year_an.index.year,year_an[c],color=np.where(year_an[c]>=0,'#d46a3d','#277c9e'),width=.85)
            axes[1].set_ylabel(f'Annual anomaly ({annual_unit(c)})')
        axes[1].axhline(0,color='black',lw=.7)
        axes[1].set_xlabel('Year')
        fig.suptitle(c+f' | baseline {args.baseline_start[:4]}–{args.baseline_end[:4]}')
        fig.tight_layout(); fig.savefig(out/'figures'/f'{c}.png',dpi=150,bbox_inches='tight'); plt.close(fig)
    shares = precipitation_shares(d)
    share_figures = []
    if not shares.empty:
        write(shares,'annual_precipitation_phase_percentages',out)
        share_trends = trends(shares,args.alpha)
        share_trends['slope_unit'] = 'percentage points/decade'
        write(share_trends,'precipitation_phase_mann_kendall_sen',out)
        share_an,share_z,share_clim = anomalies(shares,args.baseline_start,args.baseline_end)
        write(share_an,'annual_precipitation_phase_anomalies_percentage_points',out)
        write(share_clim,'precipitation_phase_baseline_climatology',out)
        for c in shares:
            s = shares[c].dropna(); row = share_trends.loc[c]
            fig,axes = plt.subplots(2,1,figsize=(12,7),sharex=True)
            axes[0].plot(s.index.year,s,label=c.replace('_',' '))
            axes[0].plot(s.index.year,row.sen_slope_per_year*s.index.year+row.sen_intercept,
                         color='crimson',label='Sen trend line')
            axes[0].set_ylabel('Share of total precipitation (%)'); axes[0].legend()
            axes[0].set_title(f'MK p={row.p_value:.4g}; Sen slope={row.sen_slope_per_decade:.4g} percentage points/decade')
            axes[1].bar(share_an.index.year,share_an[c],color=np.where(share_an[c]>=0,'#d46a3d','#277c9e'))
            axes[1].axhline(0,color='black',lw=.7)
            axes[1].set_ylabel('Annual anomaly (percentage points)'); axes[1].set_xlabel('Year')
            fig.suptitle(c.replace('_',' ')+f' | requested baseline {args.baseline_start[:4]}–{args.baseline_end[:4]}')
            fig.tight_layout(); fig.savefig(out/'figures'/f'{c}.png',dpi=150,bbox_inches='tight'); plt.close(fig)
            share_figures.append(c)
    notes = [
        f'Requested baseline: {args.baseline_start}–{args.baseline_end}; available baseline records: {ref.index.min().date()}–{ref.index.max().date()}. Missing baseline years are not reconstructed.',
        'Only trends and anomalies are analyzed. No detrending, forecasting, correlations, decomposition or extreme-value modeling.',
        'Original Mann–Kendall uses tie correction and continuity correction. Sen slopes use actual year spacing; 95% slope intervals assume independent observations.',
        'Seasonal MK compares the same calendar month across years and sums month-specific scores and variances. Seasonal Sen is the median of within-month pair slopes. Monthly daily flux means are used for this test.',
        'MK significance assumes independent observations; seasonal MK also omits cross-month covariance. Autocorrelation can make p-values optimistic. Annual lag-1 correlation is reported as a diagnostic, without removing trends or changing the series.',
        'Benjamini–Hochberg FDR is applied separately within each trend table. Statistical significance is reported at the configured alpha. No slope confidence interval is claimed for pooled seasonal Sen.',
        'Daily anomalies use calendar month/day climatology; annual and season-year anomalies use baseline annual and seasonal means. Monthly anomalies are omitted. Standardized anomalies divide by sample SD, not standard error. Zero SD gives missing z-scores.',
        'Rainfall and snowfall shares are 100 × annual component total / annual precipitation total. Snowfall uses water-equivalent mm. Zero precipitation or incomplete years yield missing ratios. Percentage anomalies and slopes use percentage points. Source components can differ slightly from total precipitation; shares are not forced to add to 100%.',
        'Plot units follow source column names and ERA5 parameter definitions. Existing derived-column units follow the named parent variables. Daily flux totals are mm/year; annually averaged source accumulations retain daily accumulation units.',
        'Complete observed periods only. Incomplete October 2026, annual 2026, and SON 2026 are excluded from respective period statistics. December belongs to the following DJF year.',
        'Trend tests require at least 10 complete years. Columns with insufficient complete years are listed in annual_trends_insufficient_data.csv.',
        'Known daily water fluxes are summed; other variables are averaged. Existing rolling, anomaly and binary columns are included if varying but their means are descriptive and their original definitions must be verified.',
        'Source CSV is preserved. Tiny negative water fluxes (-0.001 to 0 mm) become zero; more negative water fluxes become missing. Negative temperatures and signed water-balance values are retained.',
        'The reported flood is October 11, 2026; observations end October 5. Anomalies describe the available pre-event period, not unobserved flood-day conditions.',
        'One ERA5 point is not catchment-average rainfall or measured discharge. Runoff is a modeled depth. ERA5 snow depth is metres of water equivalent; verify export metadata.'
    ]
    pieces = ['<!doctype html><html><head><meta charset="utf-8"><title>Budhi trends and anomalies</title>',
              '<style>body{font:15px Arial;max-width:1200px;margin:30px auto;padding:20px} img{max-width:100%} table{border-collapse:collapse;font-size:12px} th,td{border:1px solid #ccc;padding:5px}</style></head><body>',
              '<h1>Budhi Gandaki: trend and anomaly analysis</h1>',
              f'<p>{len(raw):,} daily records, {index.min().date()}–{index.max().date()}; {len(d.columns)} variables. Baseline: {args.baseline_start}–{args.baseline_end}.</p>',
              '<h2>Methods and interpretation</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ul>',
              '<h2>Annual Mann–Kendall and Sen slope results</h2>',
              annual_trends.reindex(plots).to_html(float_format=lambda v:f'{v:.4g}'),
              '<h2>Trend and anomaly plots</h2>']
    for c in plots:
        pieces.append(f'<img loading="lazy" src="figures/{c}.png" alt="{html.escape(c)}">')
    pieces.append('<h2>Rainfall and snowfall percentages of total precipitation</h2>')
    for c in share_figures:
        pieces.append(f'<img loading="lazy" src="figures/{c}.png" alt="{c}">')
    pieces.append('<h2>All-variable tables</h2><ul>')
    for p in sorted((out/'tables').glob('*.csv')):
        pieces.append(f'<li><a href="tables/{p.name}">{p.stem}</a></li>')
    pieces.append('</ul><p>References: <a href="https://doi.org/10.2307/1907187">Mann (1945)</a>, '
                  '<a href="https://doi.org/10.1080/01621459.1968.10480934">Sen (1968)</a>, '
                  '<a href="https://doi.org/10.1029/WR018i001p00107">Hirsch et al. (1982), seasonal trends</a>, '
                  '<a href="https://confluence.ecmwf.int/pages/viewpage.action?pageId=669811810">ERA5 documentation</a>.</p></body></html>')
    (out/'report.html').write_text('\n'.join(pieces),encoding='utf-8')
    manifest = dict(source=str(args.input.resolve()),sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    rows=len(raw),variables=len(d.columns),missing_calendar_days=len(index.difference(raw.index)),
                    baseline=[args.baseline_start,args.baseline_end],
                    available_baseline=[str(ref.index.min().date()),str(ref.index.max().date())],alpha=args.alpha,
                    settings=vars(args),notes=notes,
                    versions={k:__import__(k).__version__ for k in ['numpy','pandas','scipy','matplotlib']})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    logging.info('Finished: %s',out/'report.html')


if __name__=='__main__':
    main()
