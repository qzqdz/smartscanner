# -*- coding: utf-8 -*-
"""Train CCRNet-v2 (configurable emb_dim). Supervised-contrastive triples with
same-class vuln pairing weight 0.8. batch=21 triples => 63 rows => InfoNCE chance
= log(62)~=log(63)~=4.14 (the acceptance floor). Tracks train loss + val macro-AUC
(must improve monotonically). Trains on split_wc GALLERY only (query held out for
kNN eval). CPU by default (GB10 unified-mem safety)."""
import sys, os, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0,'/home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde')
from model_v2 import CCRNetV2
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

ROOT='/home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde'
BEN=ROOT+'/v2/bench'; CLS=['y_ac','y_re','y_ar','y_uc']
def seedall(s=42):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)

def embed(model, tok, idx, dev, B=64):
    model.eval(); out=np.zeros((len(idx),512),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),B):
            x=torch.from_numpy(tok[idx[s:s+B]].astype(np.int64)).to(dev)
            out[s:s+B]=model.encode(x).cpu().numpy()
    return out

def val_auc(model, tok, df, dev):
    g=df[df.split_wc=='gallery'].index.to_numpy(); q=df[df.split_wc=='query'].index.to_numpy()
    Eg=embed(model,tok,g,dev); Eq=embed(model,tok,q,dev)
    Eg/=np.linalg.norm(Eg,axis=1,keepdims=True)+1e-9; Eq/=np.linalg.norm(Eq,axis=1,keepdims=True)+1e-9
    nn5=NearestNeighbors(n_neighbors=5,metric='cosine').fit(Eg); _,ind=nn5.kneighbors(Eq)
    Yg=df.loc[g,CLS].to_numpy(); Yq=df.loc[q,CLS].to_numpy()
    sc=Yg[ind].mean(1); aucs=[]
    for c in range(4):
        if Yq[:,c].sum()>0: aucs.append(roc_auc_score(Yq[:,c],sc[:,c]))
    return float(np.mean(aucs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--emb_dim',type=int,default=16)
    ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--bs',type=int,default=21)        # 63 rows -> log(63) floor
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--no_contrastive',action='store_true')
    ap.add_argument('--pair_w',type=float,default=0.8) # same-signature positive prob
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--tag',default=None)
    ap.add_argument('--steps_per_epoch',type=int,default=500)
    ap.add_argument('--warmup',type=int,default=0)   # linear LR warmup steps (0=off, back-compat)
    args=ap.parse_args(); seedall()
    torch.set_num_threads(16)
    tag=args.tag or f'v2_d{args.emb_dim}'+('_noctr' if args.no_contrastive else '')
    dev=torch.device(args.device)
    import model as M; M.DEVICE=dev
    tok=np.load(BEN+'/tokens.npy'); df=pd.read_parquet(BEN+'/sample.parquet').reset_index(drop=True)
    tr=df[df.split_wc=='gallery'].index.to_numpy()
    Ytr=df.loc[tr,CLS].to_numpy()
    sig=[tuple(r) for r in Ytr]
    from collections import defaultdict
    by_sig=defaultdict(list)
    for i,s in enumerate(sig): by_sig[s].append(tr[i])
    pos_class=defaultdict(list)           # contracts having class c positive
    for i,gi in enumerate(tr):
        for c in range(4):
            if Ytr[i,c]: pos_class[c].append(gi)
    sig_of={gi:sig[i] for i,gi in enumerate(tr)}
    rng=np.random.default_rng(42)
    def pick_pos(a):
        sa=sig_of[a]
        if rng.random()<args.pair_w:
            cand=by_sig[sa]
            if len(cand)>1: return int(rng.choice([x for x in cand if x!=a] or [a]))
        # share >=1 class
        cs=[c for c in range(4) if sa[c]]
        if cs:
            c=int(rng.choice(cs)); pool=pos_class[c]
            if len(pool)>1: return int(rng.choice([x for x in pool if x!=a] or [a]))
        return int(a)   # self -> dropout aug
    def pick_neg(a):
        sa=set(c for c in range(4) if sig_of[a][c])
        for _ in range(8):
            b=int(rng.choice(tr)); sb=set(c for c in range(4) if sig_of[b][c])
            if not (sa & sb): return b
        return int(rng.choice(tr))

    model=CCRNetV2(50265, emb_dim=args.emb_dim).to(dev)
    # no-contrastive control: SAME encoder, supervised multi-label BCE head instead of the
    # InfoNCE vuln-pairing objective. Holds label EXPOSURE constant (encoder still sees all
    # 4 vuln labels) but removes the contrastive MECHANISM -> isolates what the contrastive
    # objective contributes to the retrieval space, vs mere label access. (The model_v2
    # use_contrastive=False branch is zero-grad; we do NOT use it -- that would be untrained.)
    head=nn.Linear(512,4).to(dev) if args.no_contrastive else None
    params=list(model.parameters())+(list(head.parameters()) if head is not None else [])
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=1e-4)
    gstep=0
    floor=np.log(63); hist=[]; best=-1
    for ep in range(args.epochs):
        model.train(); order=rng.permutation(tr); losses=[]
        n_steps=min(args.steps_per_epoch, len(order)//args.bs)
        for st in range(n_steps):
            if args.warmup>0:
                for pg in opt.param_groups: pg['lr']=args.lr*min(1.0,(gstep+1)/args.warmup)
            gstep+=1
            anchors=order[st*args.bs:(st+1)*args.bs]
            rows=[]
            for a in anchors:
                rows+= [a, pick_pos(a), pick_neg(a)]
            rows=np.array(rows)
            x=torch.from_numpy(tok[rows].astype(np.int64)).to(dev)
            if args.no_contrastive:
                rep=model.encode(x)
                ylab=torch.from_numpy(df.loc[rows,CLS].to_numpy().astype(np.float32)).to(dev)
                loss=nn.functional.binary_cross_entropy_with_logits(head(rep),ylab)
            else:
                loss,_=model(x, use_contrastive=True)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
            if st%50==0:
                print(f'[{tag}] ep{ep} st{st}/{n_steps} loss {np.mean(losses[-50:]):.4f} (floor {floor:.3f})',flush=True)
        va=val_auc(model,tok,df,dev)
        hist.append({'epoch':ep,'loss':float(np.mean(losses)),'val_auc':va})
        print(f'[{tag}] EPOCH{ep} mean_loss {np.mean(losses):.4f} val_macroAUC {va:.4f}',flush=True)
        if va>best:
            best=va; torch.save(model.state_dict(), BEN+f'/ckpt_{tag}.pt')
    # final embed all 16k with best ckpt, save
    model.load_state_dict(torch.load(BEN+f'/ckpt_{tag}.pt',map_location=dev))
    allidx=np.arange(len(tok)); E=embed(model,tok,allidx,dev)
    np.save(BEN+f'/emb_{tag}.npy',E)
    json.dump({'tag':tag,'hist':hist,'best_val_auc':best,'floor':float(floor),
               'final_loss':hist[-1]['loss'],'monotonic':all(hist[i]['val_auc']<=hist[i+1]['val_auc']+1e-4 for i in range(len(hist)-1))},
              open(ROOT+f'/v2/artifacts/train_{tag}.json','w'),indent=2)
    print(f'[{tag}] DONE best_val_auc {best:.4f} emb saved emb_{tag}.npy')

if __name__=='__main__': main()
