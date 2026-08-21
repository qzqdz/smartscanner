# -*- coding: utf-8 -*-
"""Build durable 4-class label table for all SASC contracts (dirty/Slither labels).
4 vuln classes (OVR multilabel): access_control, reentrancy, arithmetic, unchecked_calls.
'safe' = all 4 bits zero (all-negative row per advisor). No source stored (memory guard).
Out: v2/bench/labels.parquet  +  prints prevalence.
"""
import os, json, glob
import pandas as pd

ROOT = '/home/qzqdz/Desktop/project/smart-contract-security'
SASC = ROOT + '/datasets/sasc'
OUT  = ROOT + '/smartscanner/code_icde/v2/bench/labels.parquet'

MAP = json.load(open(SASC + '/data/label_mappings.json'))
# our 4 OVR classes (hyphen form from mapping)
FOUR = {'access-control':'ac', 'reentrancy':'re', 'arithmetic':'ar', 'unchecked-calls':'uc'}

def split_map():
    df = pd.read_csv(SASC + '/data/big-splits.csv')
    return {str(a).lower(): s for a, s in zip(df['contracts'], df['split'])}

def checks(results_field):
    try:
        r = results_field if isinstance(results_field, dict) else json.loads(results_field)
        det = (r.get('results') or {}).get('detectors') or []
        return sorted({d.get('check') for d in det if d.get('check')})
    except Exception:
        return []

smap = split_map()
rows = []
files = sorted(glob.glob(SASC + '/data/raw/*.parquet'))
for fi, f in enumerate(files):
    df = pd.read_parquet(f, columns=['contracts', 'results'])
    for addr, res in zip(df['contracts'], df['results']):
        a = str(addr).lower()
        ck = checks(res)
        bits = {'ac':0,'re':0,'ar':0,'uc':0}
        for c in ck:
            m = MAP.get(c)
            if m in FOUR:
                bits[FOUR[m]] = 1
        rows.append({
            'addr': a,
            'split': smap.get(a, 'none'),
            'y_ac': bits['ac'], 'y_re': bits['re'], 'y_ar': bits['ar'], 'y_uc': bits['uc'],
            'n_det': len(ck),
            'has4': int(any(bits.values())),
            'dets': '|'.join(ck),
        })
    print(f'  parsed {f.split("/")[-1]}  cum={len(rows)}', flush=True)

out = pd.DataFrame(rows)
out.to_parquet(OUT, index=False)
print('\n=== written', OUT, 'rows', len(out))
print('=== 4-class prevalence (OVR positives) ===')
for k,name in [('y_ac','access_control'),('y_re','reentrancy'),('y_ar','arithmetic'),('y_uc','unchecked_calls')]:
    n = int(out[k].sum()); print(f'  {name:16s} {n:7d}  {100*n/len(out):5.2f}%')
allneg = int((out['has4']==0).sum())
trulyclean = int(((out['has4']==0)&(out['n_det']==0)).sum())
print(f'  {"all-negative(safe)":16s} {allneg:7d}  {100*allneg/len(out):5.2f}%   (of which 0-detector: {trulyclean})')
print('=== split x has4 ===')
print(out.groupby('split')['has4'].agg(['count','sum']))
