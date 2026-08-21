# -*- coding: utf-8 -*-
"""§3.2 DISCRIMINATING clean-vs-dirty control (advisor-corrected 2026-08-19).

Trains the SAME BCE encoder (§3.3 --no_contrastive recipe) on the SAME contract
set (in3600 ∩ split_wc==gallery ∩ relabeled) under two label sources: DIRTY (Slither)
vs CLEAN (LLM exp15). Both arms eval kNN on in3600∩query∩relabeled against CLEAN GT,
with CLEAN gallery labels voting in BOTH arms -> only the trained EMBEDDING differs.
n held constant between arms (same rows) so sample size is not a confound.

This is NOT the eval-side label swap in p0_gate_clean4.py (that one is non-discriminating:
a drop is predicted under both hypotheses). Here only the training label source varies.

Run at FULL 3600 (partial-1199 per-class positives too thin). CPU-only.
"""
import sys, os, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
ROOT='/home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde'
sys.path.insert(0, ROOT)
from model_v2 import CCRNetV2
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
BEN=ROOT+'/v2/bench'; CLS=['y_ac','y_re','y_ar','y_uc']
KMAP={'access_control':'y_ac','reentrancy':'y_re','arithmetic':'y_ar','unchecked_calls':'y_uc'}

def seedall(s=42):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_clean(path):
    m={}
    for ln in open(path):
        ln=ln.strip()
        if not ln: continue
        try: r=json.loads(ln)
        except: continue
        a=(r.get('addr') or '').lower()
        lab=r.get('labels') or {}
        if not a or not lab: continue
        m[a]={KMAP[k]:int(bool(lab.get(k))) for k in KMAP}   # last-write-wins
    return m

def embed(model, tok, idx, dev, B=64):
    model.eval(); out=np.zeros((len(idx),512),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),B):
            x=torch.from_numpy(tok[idx[s:s+B]].astype(np.int64)).to(dev)
            out[s:s+B]=model.encode(x).cpu().numpy()
    return out

def train_bce(tok, tr_idx, Ytr, emb_dim, dev, epochs, bs, lr, seed):
    seedall(seed); torch.set_num_threads(16)
    model=CCRNetV2(50265, emb_dim=emb_dim).to(dev); head=nn.Linear(512,4).to(dev)
    opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=lr, weight_decay=1e-4)
    rng=np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); order=rng.permutation(len(tr_idx)); losses=[]
        for st in range(len(order)//bs):
            b=order[st*bs:(st+1)*bs]; rows=tr_idx[b]
            x=torch.from_numpy(tok[rows].astype(np.int64)).to(dev)
            y=torch.from_numpy(Ytr[b].astype(np.float32)).to(dev)
            loss=nn.functional.binary_cross_entropy_with_logits(head(model.encode(x)), y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        print(f'    ep{ep} loss {np.mean(losses):.4f}', flush=True)
    return model

def knn_eval(Eg, Ygal_clean, Eq, Yq_clean, k=5):
    Eg=Eg/(np.linalg.norm(Eg,axis=1,keepdims=True)+1e-9)
    Eq=Eq/(np.linalg.norm(Eq,axis=1,keepdims=True)+1e-9)
    nn5=NearestNeighbors(n_neighbors=min(k,len(Eg)),metric='cosine').fit(Eg); _,ind=nn5.kneighbors(Eq)
    sc=Ygal_clean[ind].mean(1)   # gallery CLEAN labels vote in BOTH arms
    pred=(sc>=0.5).astype(int); f1s=[]; aucs=[]; per={}
    for c in range(4):
        yt=Yq_clean[:,c]
        f1=f1_score(yt,pred[:,c],zero_division=0)
        au=roc_auc_score(yt,sc[:,c]) if yt.sum()>0 and yt.sum()<len(yt) else float('nan')
        f1s.append(f1)
        if not np.isnan(au): aucs.append(au)
        per[CLS[c]]={'P':round(precision_score(yt,pred[:,c],zero_division=0),4),
                     'R':round(recall_score(yt,pred[:,c],zero_division=0),4),
                     'F1':round(f1,4),'AUC':round(au,4) if not np.isnan(au) else None,'pos':int(yt.sum())}
    return {'macro_F1':round(float(np.mean(f1s)),4),'macro_AUC':round(float(np.mean(aucs)),4) if aucs else None,
            'ac_recall':per['y_ac']['R'],'per_class':per}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--clean', default='clean3600/clean_labels.jsonl')
    ap.add_argument('--emb_dim', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    dev=torch.device('cpu')
    import model as M; M.DEVICE=dev
    tok=np.load(BEN+'/tokens.npy')
    df=pd.read_parquet(BEN+'/sample.parquet')   # default RangeIndex == tok row
    df['addr_l']=df['addr'].str.lower()
    clean=load_clean(args.clean)
    df['relab']=df['addr_l'].isin(clean.keys())
    n=int(df['relab'].sum())
    print(f'[p32] relabeled={n}  in3600={int(df.in3600.sum())}')
    if args.final:
        assert n>=3528, f'not final-ready: {n}<3528'
    # matched rows: in3600 & relabeled, split by split_wc
    gal=df[(df.in3600)&(df.relab)&(df.split_wc=='gallery')]
    qry=df[(df.in3600)&(df.relab)&(df.split_wc=='query')]
    print(f'[p32] matched gallery={len(gal)} query={len(qry)}')
    def clean_mat(rows):
        return np.array([[clean[a][c] for c in CLS] for a in rows['addr_l']], dtype=int)
    Ygal_clean=clean_mat(gal); Yq_clean=clean_mat(qry)
    Ygal_dirty=gal[CLS].to_numpy().astype(int)
    Yq_dirty=qry[CLS].to_numpy().astype(int)
    tr_idx=gal.index.to_numpy(); q_idx=qry.index.to_numpy()
    res={}; grid={}   # advisor #2: 2x2 = arm(embedding) x eval-GT(clean|dirty), clean gallery votes ALL cells
    for arm,Ytr in [('dirty',Ygal_dirty),('clean',Ygal_clean)]:
        print(f'[p32] train arm={arm} (n={len(tr_idx)})')
        model=train_bce(tok, tr_idx, Ytr, args.emb_dim, dev, args.epochs, args.bs, args.lr, seed=args.seed)
        Eg=embed(model,tok,tr_idx,dev); Eq=embed(model,tok,q_idx,dev)
        r_c=knn_eval(Eg, Ygal_clean, Eq, Yq_clean)   # reported cell (vs CLEAN GT)
        r_d=knn_eval(Eg, Ygal_dirty, Eq, Yq_dirty)   # confound probe: DIRTY votes + DIRTY GT (matched) -> well-posed within-column task
        np.save(BEN+f'/p32_Eg_{arm}_s{args.seed}.npy',Eg); np.save(BEN+f'/p32_Eq_{arm}_s{args.seed}.npy',Eq)
        res[arm]=r_c; grid[arm]={'vs_clean':r_c,'vs_dirty':r_d}
        print(f'    arm={arm}: vsCLEAN F1={r_c["macro_F1"]} AUC={r_c["macro_AUC"]} acR={r_c["ac_recall"]} | vsDIRTY F1={r_d["macro_F1"]} AUC={r_d["macro_AUC"]}')
    delta={k:(None if res['clean'][k] is None or res['dirty'][k] is None else round(res['clean'][k]-res['dirty'][k],4))
           for k in ['macro_F1','macro_AUC','ac_recall']}
    def cell(a,g,m): return grid[a][g][m]
    grid_delta={g:{m:(None if cell('clean',g,m) is None or cell('dirty',g,m) is None else round(cell('clean',g,m)-cell('dirty',g,m),4)) for m in ['macro_F1','macro_AUC','ac_recall']} for g in ['vs_clean','vs_dirty']}
    genuine=(grid_delta['vs_clean']['macro_AUC'] or 0)>0 and (grid_delta['vs_dirty']['macro_AUC'] or 0)>0
    out={'n_relabeled':n,'gallery':len(gal),'query':len(qry),'partial':not args.final,
         'dirty_arm':res['dirty'],'clean_arm':res['clean'],'clean_minus_dirty':delta,
         'grid_2x2':grid,'grid_clean_minus_dirty':grid_delta,'confound_verdict':('genuine_vuln_aware' if genuine else 'alignment_or_mixed'),
         'design':'matched-n train-side; only training label source varies; clean gallery votes both arms; 2x2 arm x eval-GT'}
    out['seed']=args.seed
    tag=('' if args.final else '_partial')+(f'_s{args.seed}' if args.seed!=42 else '')
    fp=ROOT+f'/v2/artifacts/p32_train_control{tag}.json'
    json.dump(out, open(fp,'w'), indent=2)
    print(f'[p32] clean-minus-dirty {delta}')
    print(f'[p32] wrote {fp}')

if __name__=='__main__': main()
