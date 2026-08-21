# -*- coding: utf-8 -*-
"""Recluster the 16k benchmark with VERIFIED near-clone edges (fix s1 false-merge).
LSH banding -> candidate pairs -> keep edge only if minhash-estimated Jaccard>=TAU
-> connected components (single-linkage over REAL edges). Then rebuild splits.
tokens.npy row order preserved (we keep sample.parquet order).
"""
import glob, re, random, json
import numpy as np, pandas as pd, xxhash

ROOT='/home/qzqdz/Desktop/project/smart-contract-security'; SASC=ROOT+'/datasets/sasc'
BEN=ROOT+'/smartscanner/code_icde/v2/bench'
SEED=42; TAU=0.7
NPERM,BANDS,ROWS=64,8,8; PRIME=(1<<61)-1
s=pd.read_parquet(BEN+'/sample.parquet'); addrs=list(s['addr']); keep=set(addrs)

src={}
for f in sorted(glob.glob(SASC+'/data/raw/*.parquet')):
    df=pd.read_parquet(f,columns=['contracts','source_code'])
    for a,c in zip(df['contracts'],df['source_code']):
        al=str(a).lower()
        if al in keep and al not in src: src[al]=c or ''
rnd=random.Random(SEED)
AB=[(rnd.randint(1,PRIME-1),rnd.randint(0,PRIME-1)) for _ in range(NPERM)]
def sh(code,k=5):
    code=re.sub(r'/\*.*?\*/','',code,flags=re.S)   # block comments (multiline)
    code=re.sub(r'//[^\n]*','',code)               # line comments (to EOL only)
    t=re.findall(r'\w+|[^\s\w]',code)
    if len(t)<k: return {' '.join(t)} if t else set()
    return {' '.join(t[i:i+k]) for i in range(len(t)-k+1)}
def mh(shs):
    if not shs: return None                 # degenerate -> no clone edges (singleton)
    base=[xxhash.xxh64(x.encode()).intdigest() for x in shs]
    return tuple(min((a*h+b)%PRIME for h in base) for a,b in AB)
sig={}
for i,a in enumerate(addrs):
    sig[a]=mh(sh(src.get(a,'')))
    if (i+1)%3000==0: print('mh',i+1,flush=True)
ndeg=sum(1 for a in addrs if sig[a] is None)
print('degenerate(empty-shingle) contracts:',ndeg,flush=True)

# LSH candidate pairs (only among non-degenerate)
cand=set()
for band in range(BANDS):
    buckets={}
    for a in addrs:
        v=sig[a]
        if v is None: continue
        buckets.setdefault((band,)+tuple(v[band*ROWS:(band+1)*ROWS]),[]).append(a)
    for grp in buckets.values():
        if len(grp)<2: continue
        # cap huge false buckets: pair each to first only (verification will drop false)
        if len(grp)>200:
            for o in grp[1:]: cand.add((grp[0],o))
        else:
            for i in range(len(grp)):
                for j in range(i+1,len(grp)): cand.add((grp[i],grp[j]))
print('candidate pairs',len(cand),flush=True)
# verify by minhash-estimated Jaccard = fraction of matching sig positions
parent={a:a for a in addrs}
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(x,y):
    rx,ry=find(x),find(y)
    if rx!=ry: parent[ry]=rx
kept=0
for x,y in cand:
    vx,vy=sig[x],sig[y]
    est=sum(1 for p in range(NPERM) if vx[p]==vy[p])/NPERM
    if est>=TAU: union(x,y); kept+=1
print('verified edges kept',kept,flush=True)
comp={}
for a in addrs: comp.setdefault(find(a),[]).append(a)
a2c={}; 
for cid,(rep,mem) in enumerate(comp.items()):
    for m in mem: a2c[m]=cid
s['cluster']=[a2c[a] for a in addrs]
sz=pd.Series([len(m) for m in comp.values()])
print('clusters',len(comp),'near_clone_rate',round(1-len(comp)/len(addrs),4),
      'top sizes',sorted(sz,reverse=True)[:8],'singletons',int((sz==1).sum()))

# ---- splits ----
r=np.random.RandomState(SEED)
s['split_wc']=np.where(r.rand(len(s))<0.20,'query','gallery')          # per-contract, clones straddle
# de-clone: one representative per cluster, then 80/20 split of reps
reps=[mem[0] for mem in comp.values()]
rr=random.Random(SEED); rr.shuffle(reps)
nq=int(round(0.20*len(reps))); qreps=set(reps[:nq])
rep_of=set(reps)
s['is_rep']=s['addr'].isin(rep_of)
s['split_dc']=np.where(s['addr'].isin(qreps),'query',np.where(s['is_rep'],'gallery','none'))
s.to_parquet(BEN+'/sample.parquet',index=False)
json.dump({'N':len(s),'clusters':len(comp),'near_clone_rate':round(1-len(comp)/len(addrs),4),
           'degenerate':ndeg,'verified_edges':kept,'TAU':TAU,
           'q_wc':int((s.split_wc=='query').sum()),
           'declone_reps':len(reps),'declone_q':len(qreps),'declone_g':len(reps)-len(qreps)},
          open(BEN+'/clone_report.json','w'),indent=2)
print('\n=== per-class query positives ===')
for sp in ['split_wc','split_dc']:
    q=s[s[sp]=='query']; g=s[s[sp]=='gallery']
    print(f'{sp}: gallery={len(g)} query={len(q)}')
    for k,nm in [('y_ac','ac'),('y_re','re'),('y_ar','ar'),('y_uc','uc')]:
        print(f'   {nm} q-pos={int(q[k].sum())} g-pos={int(g[k].sum())}')
