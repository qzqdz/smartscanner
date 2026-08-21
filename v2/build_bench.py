# -*- coding: utf-8 -*-
"""Build the vuln-aware kNN benchmark (dirty/Slither labels), clone-clustered, tokenized.
Outputs under v2/bench/:
  sample.parquet  addr,y_ac,y_re,y_ar,y_uc,split_orig,cluster,split_wc,split_dc,in3600,tok_len
  tokens.npy      int32 [N,4096] head2048+tail2048, pad=1
  manifest_3600.json  addrs for the clean (exp15) arm  (same split assignment)
  clone_report.json
"""
import os, re, json, glob, random
import numpy as np, pandas as pd, xxhash
from transformers import AutoTokenizer

ROOT = '/home/qzqdz/Desktop/project/smart-contract-security'
SASC = ROOT + '/datasets/sasc'
V2   = ROOT + '/smartscanner/code_icde/v2'
BEN  = V2 + '/bench'
TOK  = ROOT + '/smartscanner/code_icde/SC_model_big_long_resnet_1d_24000'
SEED = 42; N = 16000; HEAD = TAIL = 2048; MAXLEN = HEAD + TAIL
random.seed(SEED); np.random.seed(SEED)

lab = pd.read_parquet(BEN + '/labels.parquet')
# sample N at natural prevalence
idx = np.random.RandomState(SEED).permutation(len(lab))[:N]
samp = lab.iloc[np.sort(idx)].reset_index(drop=True)
keep = set(samp['addr'])
print(f'[bench] sampled {len(samp)} at natural prevalence', flush=True)

# pull source for kept addrs from parquets (one file at a time = memory guard)
src = {}
for f in sorted(glob.glob(SASC + '/data/raw/*.parquet')):
    df = pd.read_parquet(f, columns=['contracts', 'source_code'])
    for a, c in zip(df['contracts'], df['source_code']):
        al = str(a).lower()
        if al in keep and al not in src:
            src[al] = c or ''
    print(f'  src pulled {len(src)}/{len(keep)} after {f.split("/")[-1]}', flush=True)
samp = samp[samp['addr'].isin(src)].reset_index(drop=True)
print(f'[bench] with source: {len(samp)}', flush=True)

# ---- MinHash clone clustering (reuse s1_dedup scheme) ----
NPERM, BANDS, ROWS = 64, 8, 8; PRIME = (1<<61)-1
rnd = random.Random(SEED)
AB = [(rnd.randint(1,PRIME-1), rnd.randint(0,PRIME-1)) for _ in range(NPERM)]
def shingles(code, k=5):
    toks = re.findall(r'\w+|[^\s\w]', re.sub(r'//.*|/\*.*?\*/','',code, flags=re.S))
    if len(toks) < k: return {' '.join(toks)} if toks else set()
    return {' '.join(toks[i:i+k]) for i in range(len(toks)-k+1)}
def minhash(sh):
    if not sh: return tuple([0]*NPERM)
    base = [xxhash.xxh64(s.encode()).intdigest() for s in sh]
    return tuple(min((a*h+b)%PRIME for h in base) for a,b in AB)
sigs = {}
for i,a in enumerate(samp['addr']):
    sigs[a] = minhash(shingles(src[a]))
    if (i+1)%2000==0: print(f'  minhash {i+1}', flush=True)
parent = {}
def find(x):
    parent.setdefault(x,x)
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(x,y):
    rx,ry=find(x),find(y)
    if rx!=ry: parent[ry]=rx
for a in sigs: find(a)
for band in range(BANDS):
    buckets={}
    for a,sig in sigs.items():
        buckets.setdefault((band,)+tuple(sig[band*ROWS:(band+1)*ROWS]),[]).append(a)
    for grp in buckets.values():
        for o in grp[1:]: union(grp[0],o)
cl = {}
for a in sigs: cl.setdefault(find(a),[]).append(a)
addr2c = {}
for cid,(rep,mem) in enumerate(cl.items()):
    for m in mem: addr2c[m]=cid
samp['cluster'] = samp['addr'].map(addr2c)
nclust = len(cl); near_rate = 1 - nclust/len(samp)
print(f'[bench] clusters={nclust}  near_clone_rate={near_rate:.3f}', flush=True)

# ---- splits ----
# with-clone: per-contract 80/20 (clones may straddle)
r = np.random.RandomState(SEED)
samp['split_wc'] = np.where(r.rand(len(samp)) < 0.20, 'query', 'gallery')
# de-clone: whole cluster to one side, ~20% clusters -> query
cids = list(cl.keys()); rnd2 = random.Random(SEED); rnd2.shuffle(cids)
qsize = 0; target = 0.20*len(samp); qclust=set()
for rep in cids:
    if qsize >= target: break
    qclust.add(addr2c[cl[rep][0]]); qsize += len(cl[rep])
samp['split_dc'] = np.where(samp['cluster'].isin(qclust), 'query', 'gallery')

# nested 3600 for clean arm
r3 = np.random.RandomState(SEED)
i3 = r3.permutation(len(samp))[:3600]
samp['in3600'] = False
samp.loc[samp.index[np.sort(i3)], 'in3600'] = True

# ---- tokenize head+tail ----
tk = AutoTokenizer.from_pretrained(TOK)
PAD = tk.pad_token_id if tk.pad_token_id is not None else 1
toks = np.full((len(samp), MAXLEN), PAD, dtype=np.int32)
tlen = np.zeros(len(samp), dtype=np.int32)
addrs = list(samp['addr'])
B = 400
for s in range(0, len(addrs), B):
    chunk = addrs[s:s+B]
    enc = tk([src[a][:200000] for a in chunk], add_special_tokens=True,
             truncation=False)['input_ids']
    for j, ids in enumerate(enc):
        if len(ids) > MAXLEN:
            ids = ids[:HEAD] + ids[-TAIL:]
        toks[s+j, :len(ids)] = ids
        tlen[s+j] = len(ids)
    print(f'  tok {min(s+B,len(addrs))}/{len(addrs)}', flush=True)
samp['tok_len'] = tlen

os.makedirs(BEN, exist_ok=True)
np.save(BEN + '/tokens.npy', toks)
samp.drop(columns=['dets']).to_parquet(BEN + '/sample.parquet', index=False)
json.dump({'addrs': list(samp.loc[samp['in3600'],'addr'])}, open(BEN+'/manifest_3600.json','w'))
json.dump({'N':len(samp),'clusters':nclust,'near_clone_rate':round(near_rate,4),
           'q_wc':int((samp['split_wc']=='query').sum()),
           'q_dc':int((samp['split_dc']=='query').sum())},
          open(BEN+'/clone_report.json','w'), indent=2)
print('\n=== bench built ===')
for sp in ['split_wc','split_dc']:
    q = samp[samp[sp]=='query']; g = samp[samp[sp]=='gallery']
    print(f'{sp}: gallery={len(g)} query={len(q)}')
    for k,nm in [('y_ac','ac'),('y_re','re'),('y_ar','ar'),('y_uc','uc')]:
        print(f'   {nm} query-pos={int(q[k].sum())} gallery-pos={int(g[k].sum())}')
print('median tok_len', int(np.median(tlen)), 'frac truncated', round(float((tlen>=MAXLEN).mean()),3))
