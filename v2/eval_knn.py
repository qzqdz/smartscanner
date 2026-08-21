# -*- coding: utf-8 -*-
"""Shared vuln-aware kNN eval harness. 4-class OVR multilabel.
Given per-contract embeddings (aligned to bench/sample.parquet row order),
fit kNN on gallery, predict each query's 4-class multi-hot by neighbor vote.
Metrics: per-class P/R/F1@0.5 + macro-F1, per-class AUC + macro-AUC, ac-recall.
Bootstrap CIs by resampling QUERY CLUSTERS (clone-grouped resampling unit).
Usage: from eval_knn import evaluate; evaluate(emb, split_col, k=5)
"""
import numpy as np, pandas as pd, json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

ROOT='/home/qzqdz/Desktop/project/smart-contract-security'
BEN=ROOT+'/smartscanner/code_icde/v2/bench'
CLS=['y_ac','y_re','y_ar','y_uc']; NM=['access_control','reentrancy','arithmetic','unchecked_calls']

def _l2(x):
    n=np.linalg.norm(x,axis=1,keepdims=True); n[n==0]=1; return x/n

def _predict(emb, samp, split_col, k):
    g=samp[samp[split_col]=='gallery']; q=samp[samp[split_col]=='query']
    Xg=_l2(emb[g.index.values].astype(np.float32)); Xq=_l2(emb[q.index.values].astype(np.float32))
    Yg=g[CLS].values.astype(np.float32)
    nn=NearestNeighbors(n_neighbors=min(k,len(g)),metric='cosine').fit(Xg)
    _,idx=nn.kneighbors(Xq)
    scores=Yg[idx].mean(axis=1)            # (nq,4) soft = frac of k neighbors positive
    return q, scores, q[CLS].values.astype(int)

def _metrics(scores, ytrue):
    out={}
    f1s=[]; aucs=[]
    for j,nm in enumerate(NM):
        yt=ytrue[:,j]; sc=scores[:,j]; pred=(sc>=0.5).astype(int)
        f1=f1_score(yt,pred,zero_division=0)
        pr=precision_score(yt,pred,zero_division=0); rc=recall_score(yt,pred,zero_division=0)
        auc=roc_auc_score(yt,sc) if (yt.sum()>0 and yt.sum()<len(yt)) else float('nan')
        out[nm]={'P':round(pr,4),'R':round(rc,4),'F1':round(f1,4),'AUC':round(auc,4),'pos':int(yt.sum())}
        f1s.append(f1); aucs.append(auc)
    out['macro_F1']=round(float(np.mean(f1s)),4)
    out['macro_AUC']=round(float(np.nanmean(aucs)),4)
    out['ac_recall']=out['access_control']['R']
    return out

def evaluate(emb, split_col='split_wc', k=5, B=1000, seed=42, tag=''):
    samp=pd.read_parquet(BEN+'/sample.parquet')
    q, scores, ytrue = _predict(emb, samp, split_col, k)
    base=_metrics(scores, ytrue)
    # bootstrap over query clusters
    clusters=q['cluster'].values
    uniq=np.unique(clusters); rng=np.random.RandomState(seed)
    mF1=[]; mAUC=[]; acR=[]
    cl2rows={c:np.where(clusters==c)[0] for c in uniq}
    for _ in range(B):
        pick=rng.choice(uniq,size=len(uniq),replace=True)
        rows=np.concatenate([cl2rows[c] for c in pick])
        m=_metrics(scores[rows], ytrue[rows])
        mF1.append(m['macro_F1']); mAUC.append(m['macro_AUC']); acR.append(m['ac_recall'])
    def ci(a): a=np.array(a); a=a[~np.isnan(a)]; return [round(float(np.percentile(a,2.5)),4),round(float(np.percentile(a,97.5)),4)]
    base['ci_macro_F1']=ci(mF1); base['ci_macro_AUC']=ci(mAUC); base['ci_ac_recall']=ci(acR)
    base['split']=split_col; base['k']=k; base['n_query']=int(len(q)); base['tag']=tag
    return base

if __name__=='__main__':
    import sys
    emb=np.load(sys.argv[1]); tag=sys.argv[2] if len(sys.argv)>2 else ''
    for sc in ['split_wc','split_dc']:
        r=evaluate(emb,sc,tag=tag)
        print(json.dumps({k:v for k,v in r.items() if k in
              ('tag','split','n_query','macro_F1','ci_macro_F1','macro_AUC','ci_macro_AUC','ac_recall','ci_ac_recall')}))
