"""Verify units, annual precipitation shares, output links and plot coverage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import budhi_analysis as b


def main():
    idx = pd.date_range('2024-01-01','2025-02-01')
    d = pd.DataFrame({'precipitation_mm_day':10.,'rainfall_mm_day':6.,
                      'snowfall_mm_day_water_equivalent':4.},index=idx)
    r = b.precipitation_shares(d)
    assert np.isclose(r.at[pd.Timestamp('2024-01-01'),'rainfall_percent_of_ppt'],60)
    assert np.isclose(r.at[pd.Timestamp('2024-01-01'),'snowfall_percent_of_ppt'],40)
    assert r.loc['2025-01-01'].isna().all()
    d.loc['2024-01-01','rainfall_mm_day'] = np.nan
    assert b.precipitation_shares(d).loc['2024-01-01'].isna().all()
    d.loc[:,:] = 0.
    assert b.precipitation_shares(d).isna().all().all()
    print('PASS: component percentages, shared valid days, incomplete years, zero precipitation')
    out = Path('trend_anomaly_output')
    units = pd.read_csv(out/'tables/variable_aggregation_rules.csv',index_col=0)
    assert units.daily_source_unit.notna().all()
    assert units.annual_plot_unit.notna().all()
    assert all((out/'figures'/f'{c}.png').exists() for c in units.index)
    shares = pd.read_csv(out/'tables/annual_precipitation_phase_percentages.csv',index_col=0)
    values = pd.read_csv(out/'tables/annual_values.csv',index_col=0)
    for col,source in [('rainfall_percent_of_ppt','rainfall_mm_day'),
                       ('snowfall_percent_of_ppt','snowfall_mm_day_water_equivalent')]:
        expected = 100*values[source]/values.precipitation_mm_day.where(values.precipitation_mm_day>0)
        assert np.allclose(shares[col],expected,equal_nan=True)
        assert (out/'figures'/f'{col}.png').exists()
    text = (out/'report.html').read_text(encoding='utf-8')
    q = chr(34)
    links = [s.split(q)[0] for marker in ['href='+q,'src='+q] for s in text.split(marker)[1:]]
    assert all(x.startswith('https:') or (out/x).exists() for x in links)
    assert not list((out/'tables').glob('monthly*anomal*.csv'))
    manifest = json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['baseline']==['1945-01-01','1975-12-31']
    print(f'PASS: {len(units)} variable plots, two precipitation-share plots, units, ratios, report assets and baseline')


if __name__=='__main__':
    main()
