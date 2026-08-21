# -*- coding: utf-8 -*-
"""§3.2 DISCRIMINATING clean-vs-dirty control -- CONTRASTIVE objective (advisor #2, 2026-08-19).

Same matched design as p32_train_control.py but trains the spec-mandated CONTRASTIVE
encoder (InfoNCE vuln-pairing, pair_w=0.8, CC 3->emb_dim, batch=21 triples) instead of BCE.

Both arms:
  - train on the SAME 2839 in3600 gallery rows (matched n, matched step budget -> same epochs)
  - the ONLY difference is the label source used to build contrastive pairs (dirty Slither vs clean LLM)
  - eval kNN on the 749 in3600 query rows vs CLEAN GT
  - CLEAN gallery labels vote in BOTH arms -> only the trained EMBEDDING differs
Report the PAIRED delta (clean - dirty), NOT absolutes vs the 16k-gallery 30-ep run.
Clean-subset numbers are NON-COMPARABLE to the 16k dirty benchmark (2839 vs 12741 gallery, 4.5x).
This answers "does clean-label training improve the vuln-aware space", NOT "does it clear R2 bars".
CPU-only. Gates on loss plateau (prints per-epoch loss + val macroAUC)."""
import sys, os, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import defaultdict
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
        a=(r.get('addr') or '').lower(); lab=r.get('labels') or {}
        if not a or not lab: continue
        m[a]={KMAP[k]:int(bool(lab.get(k))) for k in KMAP}
    return m

def embed(model, tok, idx, dev, B=64):
    model.eval(); out=np.zeros((len(idx),512),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),B):
            x=torch.from_numpy(tok[idx[s:s+B]].astype(np.int64)).to(dev)
            out[s:s+B]=model.encode(x).cpu().numpy()
    return out

def train_contrastive(tok, tr_idx, Ytr, emb_dim, dev, epochs, bs, lr, pair_w, warmup, seed,
                      q_idx=None, Ygal_clean=None, Yq_clean=None):
    """Contrastive triples using the ARM's labels Ytr to build same-vuln pairs. Matched budget."""
    seedall(seed); torch.set_num_threads(16)
    rng=np.random.default_rng(seed)
    tr=tr_idx; Y=Ytr
    sig=[tuple(r) for r in Y]
    by_sig=defaultdict(list)
    for i,s in enumerate(sig): by_sig[s].append(tr[i])
    pos_class=defaultdict(list)
    for i in range(len(tr)):
        for c in range(4):
            if Y[i,c]: pos_class[c].append(tr[i])
    sig_of={tr[i]:sig[i] for i in range(len(tr))}
    def pick_pos(a):
        sa=sig_of[a]
        if rng.random()<pair_w:
            cand=by_sig[sa]
            if len(cand)>1: return int(rng.choice([x for x in cand if x!=a] or [a]))
        cs=[c for c in range(4) if sa[c]]
        if cs:
            c=int(rng.choice(cs)); pool=pos_class[c]
            if len(pool)>1: return int(rng.choice([x for x in pool if x!=a] or [a]))
        return int(a)
    def pick_neg(a):
        sa=set(c for c in range(4) if sig_of[a][c])
        for _ in range(8):
            b=int(rng.choice(tr)); sb=set(c for c in range(4) if sig_of[b][c])
            if not (sa & sb): return b
        return int(rng.choice(tr))
    model=CCRNetV2(50265, emb_dim=emb_dim).to(dev)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gstep=0; hist=[]; best=-1; floor=float(np.log(63))
    for ep in range(epochs):
        model.train(); order=rng.permutation(tr); losses=[]
        nst=len(order)//bs
        for st in range(nst):
            if warmup>0:
                for pg in opt.param_groups: pg['lr']=lr*min(1.0,(gstep+1)/warmup)
            gstep+=1
            anchors=order[st*bs:(st+1)*bs]; rows=[]
            for a in anchors: rows+=[a, pick_pos(a), pick_neg(a)]
            x=torch.from_numpy(tok[np.array(rows)].astype(np.int64)).to(dev)
            loss,_=model(x, use_contrastive=True)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        va=None
        if q_idx is not None:
            Eg=embed(model,tok,tr,dev); Eq=embed(model,tok,q_idx,dev)
            va=knn_eval(Eg,Ygal_clean,Eq,Yq_clean)['macro_AUC']
        hist.append({'ep':ep,'loss':float(np.mean(losses)),'val_auc':va})
        print(f'    ep{ep} loss {np.mean(losses):.4f} (floor {floor:.3f}) val_macroAUC {va}', flush=True)
        if va is not None and va>best: best=va   # tracked for plateau-evidence logging ONLY
    # advisor #1 (2026-08-19): report FINAL-epoch model, NOT max-over-epochs on the eval set
    # (eval-set checkpoint selection biases each arm by ~its val-AUC noise range ~0.038,
    #  same magnitude as the effect). val curve retained in hist purely as plateau evidence.
    return model, hist, best

def knn_eval(Eg, Ygal_clean, Eq, Yq_clean, k=5):
    Eg=Eg/(np.linalg.norm(Eg,axis=1,keepdims=True)+1e-9)
    Eq=Eq/(np.linalg.norm(Eq,axis=1,keepdims=True)+1e-9)
    nn5=NearestNeighbors(n_neighbors=min(k,len(Eg)),metric='cosine').fit(Eg); _,ind=nn5.kneighbors(Eq)
    sc=Ygal_clean[ind].mean(1); pred=(sc>=0.5).astype(int); f1s=[]; aucs=[]; per={}
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
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--bs', type=int, default=21)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--pair_w', type=float, default=0.8)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--device', default='cpu')
    args=ap.parse_args()
    dev=torch.device(args.device)
    import model as M; M.DEVICE=dev
    tok=np.load(BEN+'/tokens.npy')
    df=pd.read_parquet(BEN+'/sample.parquet'); df['addr_l']=df['addr'].str.lower()
    clean=load_clean(args.clean); df['relab']=df['addr_l'].isin(clean.keys())
    n=int(df['relab'].sum()); print(f'[p32c] relabeled={n} in3600={int(df.in3600.sum())}')
    if args.final: assert n>=3528, f'not final-ready: {n}<3528'
    gal=df[(df.in3600)&(df.relab)&(df.split_wc=='gallery')]
    qry=df[(df.in3600)&(df.relab)&(df.split_wc=='query')]
    print(f'[p32c] matched gallery={len(gal)} query={len(qry)} | epochs={args.epochs} bs={args.bs} steps/ep={len(gal)//args.bs}')
    def clean_mat(rows): return np.array([[clean[a][c] for c in CLS] for a in rows['addr_l']], dtype=int)
    Ygal_clean=clean_mat(gal); Yq_clean=clean_mat(qry)
    Ygal_dirty=gal[CLS].to_numpy().astype(int)
    Yq_dirty=qry[CLS].to_numpy().astype(int)
    tr_idx=gal.index.to_numpy(); q_idx=qry.index.to_numpy()
    res={}; hists={}; grid={}   # grid[arm][evalGT] -> knn_eval on that GT (clean gallery votes ALWAYS)
    for arm,Ytr in [('dirty',Ygal_dirty),('clean',Ygal_clean)]:
        print(f'[p32c] train arm={arm} contrastive (n={len(tr_idx)})')
        model,hist,best=train_contrastive(tok, tr_idx, Ytr, args.emb_dim, dev, args.epochs,
                                          args.bs, args.lr, args.pair_w, args.warmup, seed=42,
                                          q_idx=q_idx, Ygal_clean=Ygal_clean, Yq_clean=Yq_clean)
        Eg=embed(model,tok,tr_idx,dev); Eq=embed(model,tok,q_idx,dev)
        # advisor #2: 2x2 = arm(embedding) x eval-GT(clean|dirty). CLEAN gallery votes in ALL cells,
        # so ONLY the embedding + which GT we score against vary -> separates a genuinely better
        # vuln-aware embedding (clean wins on BOTH GTs) from mere label-alignment (each wins own GT).
        r_c=knn_eval(Eg, Ygal_clean, Eq, Yq_clean)     # reported §3.2 cell (vs clean GT)
        r_d=knn_eval(Eg, Ygal_dirty, Eq, Yq_dirty)     # confound probe: DIRTY votes + DIRTY GT (matched)
        np.save(BEN+f'/p32c_Eg_{arm}.npy',Eg); np.save(BEN+f'/p32c_Eq_{arm}.npy',Eq)
        res[arm]=r_c; grid[arm]={'vs_clean':r_c,'vs_dirty':r_d}; hists[arm]=hist
        print(f'    arm={arm}: vsCLEAN F1={r_c["macro_F1"]} AUC={r_c["macro_AUC"]} acR={r_c["ac_recall"]} | vsDIRTY F1={r_d["macro_F1"]} AUC={r_d["macro_AUC"]} | best_val_auc={best:.4f}')
    delta={k:(None if res['clean'][k] is None or res['dirty'][k] is None else round(res['clean'][k]-res['dirty'][k],4))
           for k in ['macro_F1','macro_AUC','ac_recall']}
    # confound verdict: does clean embedding beat dirty embedding on the DIRTY GT column too?
    def cell(a,g,m): return grid[a][g][m]
    grid_delta={g:{m:(None if cell('clean',g,m) is None or cell('dirty',g,m) is None else round(cell('clean',g,m)-cell('dirty',g,m),4)) for m in ['macro_F1','macro_AUC','ac_recall']} for g in ['vs_clean','vs_dirty']}
    genuine = (grid_delta['vs_clean']['macro_AUC'] or 0)>0 and (grid_delta['vs_dirty']['macro_AUC'] or 0)>0
    out={'objective':'contrastive','n_relabeled':n,'gallery':len(gal),'query':len(qry),'partial':not args.final,
         'epochs':args.epochs,'steps_per_epoch':len(gal)//args.bs,
         'dirty_arm':res['dirty'],'clean_arm':res['clean'],'clean_minus_dirty':delta,'hist':hists,
         'grid_2x2':grid,'grid_clean_minus_dirty':grid_delta,'confound_verdict':('genuine_vuln_aware' if genuine else 'alignment_or_mixed'),
         'denominator_stamp':'gallery=2839 (matched subset), NON-COMPARABLE to 16k dirty benchmark 0.6773/0.8682/0.5258; R2 bars remain adjudicated on 16k dirty. This measures clean-vs-dirty training effect on the vuln-aware space only.',
         'design':'matched-n train-side CONTRASTIVE; only pairing label source varies; both eval vs clean; clean gallery votes both arms'}
    tag='' if args.final else '_partial'
    fp=ROOT+f'/v2/artifacts/p32c_contrastive{tag}.json'
    json.dump(out, open(fp,'w'), indent=2)
    print(f'[p32c] clean-minus-dirty {delta}')
    print(f'[p32c] wrote {fp}')

if __name__=='__main__': main()
