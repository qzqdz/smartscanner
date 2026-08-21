# -*- coding: utf-8 -*-
"""p0_gate_clean4 — clean 4-class gate for R1-clean / R3 §3.2.

Converts clean3600/clean_labels.jsonl (LLM defense-aware relabel; labels dict
access_control/reentrancy/arithmetic/unchecked_calls/safe) into clean 4-class
labels (y_ac/y_re/y_ar/y_uc), joins to bench/sample.parquet on `addr` (all clean
addrs fall inside in3600==True). Emits:

 (1) matched clean-vs-dirty label table  -> bench/clean_bench[_partial].parquet
 (2) per-class dirty->clean FLIP stats    (model-free; = Slither per-class FP, feeds C1)
 (3) §3.2 clean-vs-dirty kNN CONTROL      (fixed embedding, swap ONLY the labels,
                                           matched contracts + matched split)

Definitive benchmark requires full 3600 at <=2% error. On a partial relabel this
runs anyway and writes *_partial + tags report 'partial' (de-risk, preliminary).
Reuses eval_knn._predict/_metrics so §3.2 numbers are apples-to-apples with §3.3.
"""
import sys, os, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_knn import _predict, _metrics, CLS, NM   # CLS=y_*, NM=names, k=5 cosine

ROOT='/home/qzqdz/Desktop/project/smart-contract-security'
V2=ROOT+'/smartscanner/code_icde/v2'; BEN=V2+'/bench'
CLEANJL=V2+'/clean3600/clean_labels.jsonl'
KMAP={'access_control':'y_ac','reentrancy':'y_re','arithmetic':'y_ar','unchecked_calls':'y_uc'}

def load_clean():
    """addr -> {y_ac,y_re,y_ar,y_uc} int, last write wins (resumable/append-only)."""
    d={}
    for line in open(CLEANJL):
        line=line.strip()
        if not line: continue
        try: r=json.loads(line)
        except: continue
        a=r.get('addr') or r.get('address')
        lab=r.get('labels')
        if not a or not isinstance(lab,dict): continue
        d[a.lower()]={KMAP[k]:int(bool(lab.get(k,False))) for k in KMAP}
    return d

def flip_stats(samp_m, clean):
    """model-free: how the LLM relabel changes each class's positives on matched set."""
    out={}
    for j,c in enumerate(CLS):
        dirty=samp_m[c].values.astype(int)
        cl=np.array([clean[a][c] for a in samp_m['addr']],int)
        p10=int(((dirty==1)&(cl==0)).sum())   # dirty-positive turned clean-negative (mitigated/FP)
        p01=int(((dirty==0)&(cl==1)).sum())   # dirty-negative turned clean-positive (missed)
        dp=int((dirty==1).sum())
        cp=int((cl==1).sum())
        # NOTE: this is NOT "Slither 误报率" (denominator unresolved, see handoff §1).
        # It is the matched-subset re-annotation churn in BOTH directions.
        out[NM[j]]={'dirty_pos':dp,'clean_pos':cp,
                    'flip_1to0':p10,'flip_0to1':p01,
                    'churn_1to0_over_dirtypos':round(p10/dp,4) if dp else None,   # dirty-pos dropped
                    'churn_0to1_over_cleanpos':round(p01/cp,4) if cp else None}   # clean-pos Slither missed
    return out

def knn_labels(emb, samp, split_col):
    q,scores,ytrue=_predict(emb,samp,split_col,5)
    m=_metrics(scores,ytrue); m['n_query']=int(len(q)); m['n_gallery']=int((samp[split_col]=='gallery').sum())
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--emb',default=BEN+'/emb_v2_d16.npy',help='fixed embedding for §3.2 control')
    ap.add_argument('--final',action='store_true',help='assert >=3528 (98%% of 3600) before writing definitive clean_bench.parquet')
    args=ap.parse_args()

    clean=load_clean(); n=len(clean)
    samp=pd.read_parquet(BEN+'/sample.parquet')     # default RangeIndex == emb/token row position
    in36=int(samp['in3600'].sum())
    mask=samp['addr'].str.lower().isin(clean)
    samp_m=samp[mask].copy()                          # preserves original index -> emb alignment
    samp_m['addr']=samp_m['addr'].str.lower()
    covered=len(samp_m)
    partial = n < 3528
    print(f'[gate] clean_labels={n}  in3600={in36}  matched_in_bench={covered}  '
          f'({"PARTIAL" if partial else "FULL"})',flush=True)
    if args.final and partial:
        sys.exit(f'[gate] --final refused: only {n}/3600 relabeled (<98%). rerun when complete.')

    # (2) model-free flip stats (Slither per-class FP on matched set)
    flips=flip_stats(samp_m, clean)
    print('[gate] dirty<->clean per-class RE-ANNOTATION churn (NOT "Slither 误报率"; denom unresolved):')
    for nm,v in flips.items():
        print(f'   {nm:16s} dirty_pos={v["dirty_pos"]:4d} clean_pos={v["clean_pos"]:4d} '
              f'1->0={v["flip_1to0"]:4d}({v["churn_1to0_over_dirtypos"]}) '
              f'0->1={v["flip_0to1"]:3d}({v["churn_0to1_over_cleanpos"]})')

    # build clean-label variant of the matched table (swap ONLY y_*)
    samp_clean=samp_m.copy()
    for c in CLS:
        samp_clean[c]=[clean[a][c] for a in samp_clean['addr']]

    # (3) §3.2 clean-vs-dirty kNN control: fixed emb, matched contracts, swap labels
    emb=np.load(args.emb); emb_tag=os.path.basename(args.emb).replace('emb_','').replace('.npy','')
    ctrl={}
    for split in ['split_wc','split_dc']:
        ng=int((samp_m[split]=='gallery').sum()); nq=int((samp_m[split]=='query').sum())
        if ng<5 or nq<5:
            print(f'[gate] {split}: too few matched rows (g={ng},q={nq}) - skip'); continue
        dm=knn_labels(emb, samp_m, split)      # dirty labels
        cm=knn_labels(emb, samp_clean, split)  # clean labels
        ctrl[split]={'dirty':dm,'clean':cm}
        # DIAGNOSTIC ONLY (non-discriminating): emb trained on dirty labels, so an eval-side
        # label swap drops under BOTH hypotheses -> cannot serve as the §3.2 anti-circularity
        # control. The real §3.2 = matched-n TRAIN-side control (train BCE on dirty vs clean
        # over identical in3600∩relabeled rows, eval both vs clean). Run at full 3600.
        print(f'[gate] label-source DIAGNOSTIC (non-discriminating) {split} emb={emb_tag} g={dm["n_gallery"]} q={dm["n_query"]}')
        for src,m in (('dirty',dm),('clean',cm)):
            print(f'     {src:5s} macroF1={m["macro_F1"]:.4f} macroAUC={m["macro_AUC"]:.4f} acR={m["ac_recall"]:.4f}')

    # (1) write matched table
    outp=BEN+('/clean_bench_partial.parquet' if partial else '/clean_bench.parquet')
    keep=['addr','split_wc','split_dc','cluster','in3600']+CLS
    tbl=samp_m[keep].copy()
    for c in CLS: tbl['clean_'+c]=[clean[a][c] for a in tbl['addr']]
    tbl.to_parquet(outp)
    rep={'n_clean':n,'matched_in_bench':covered,'in3600':in36,'partial':partial,
         'emb':emb_tag,'flip_stats':flips,'label_source_diagnostic_NONDISCRIMINATING':ctrl,'out_table':outp}
    rj=V2+'/artifacts/clean_gate_report'+('_partial' if partial else '')+'.json'
    json.dump(rep,open(rj,'w'),indent=2)
    print(f'[gate] wrote {outp}\n[gate] wrote {rj}')

if __name__=='__main__': main()
