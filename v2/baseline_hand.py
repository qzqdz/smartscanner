# -*- coding: utf-8 -*-
"""Hand-crafted ~46-dim security feature baseline -> vuln-aware kNN eval."""
import glob, re, json
import numpy as np, pandas as pd
from eval_knn import evaluate
ROOT='/home/qzqdz/Desktop/project/smart-contract-security'; SASC=ROOT+'/datasets/sasc'
BEN=ROOT+'/smartscanner/code_icde/v2/bench'
s=pd.read_parquet(BEN+'/sample.parquet'); addrs=list(s['addr']); keep=set(addrs)
src={}
for f in sorted(glob.glob(SASC+'/data/raw/*.parquet')):
    df=pd.read_parquet(f,columns=['contracts','source_code'])
    for a,c in zip(df['contracts'],df['source_code']):
        al=str(a).lower()
        if al in keep and al not in src: src[al]=c or ''
PATS=[
 (r'.', 'len_chars'), (r'\n','n_lines'),
 (r'\bfunction\b','n_func'), (r'\bmodifier\b','n_modifier'),
 (r'\brequire\s*\(','n_require'), (r'\bassert\s*\(','n_assert'), (r'\brevert\b','n_revert'),
 (r'\.call\s*[{(]','n_call'), (r'\.call\s*\{\s*value','n_callvalue'),
 (r'\.send\s*\(','n_send'), (r'\.transfer\s*\(','n_transfer'),
 (r'delegatecall','n_delegatecall'), (r'selfdestruct|suicide','n_selfdestruct'),
 (r'tx\.origin','n_txorigin'), (r'msg\.sender','n_msgsender'), (r'msg\.value','n_msgvalue'),
 (r'onlyOwner','n_onlyowner'), (r'nonReentrant|ReentrancyGuard','n_nonreentrant'),
 (r'SafeMath','n_safemath'), (r'\bunchecked\s*\{','n_unchecked'),
 (r'\bfor\s*\(','n_for'), (r'\bwhile\s*\(','n_while'), (r'\bmapping\s*\(','n_mapping'),
 (r'\bpayable\b','n_payable'), (r'\bexternal\b','n_external'), (r'\bpublic\b','n_public'),
 (r'\bprivate\b','n_private'), (r'\binternal\b','n_internal'), (r'\bview\b','n_view'), (r'\bpure\b','n_pure'),
 (r'\bapprove\b','n_approve'), (r'transferFrom','n_transferfrom'), (r'\bmint\b','n_mint'),
 (r'\bwithdraw','n_withdraw'), (r'\bowner\b','n_owner'),
 (r'block\.timestamp|\bnow\b','n_timestamp'), (r'block\.number','n_blocknum'),
 (r'keccak256','n_keccak'), (r'abi\.encode','n_abiencode'), (r'\bassembly\b','n_assembly'),
 (r'\bnew\s+[A-Z]','n_new'), (r'\bimport\b','n_import'), (r'\bis\s+[A-Z]','n_inherit'),
 (r'\bconstructor\b','n_constructor'), (r'[+\-*/]','n_arithop'), (r'=>','n_arrow'),
]
def feats(code):
    v=[]
    for pat,_ in PATS: v.append(len(re.findall(pat,code)))
    # 2 pragma booleans
    pv=re.search(r'pragma\s+solidity\s+[^\n;]*',code); pv=pv.group(0) if pv else ''
    v.append(1 if re.search(r'0\.[8-9]|0\.\d\d',pv) else 0)
    v.append(1 if re.search(r'0\.[4-7]',pv) else 0)
    return v
X=np.array([feats(src.get(a,'')) for a in addrs],dtype=np.float64)
# log1p the count columns (first len(PATS)), leave 2 bools
X[:,:len(PATS)]=np.log1p(X[:,:len(PATS)])
mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; X=(X-mu)/sd
np.save(BEN+'/emb_hand.npy',X.astype(np.float32))
print('hand features dim',X.shape[1])
res={}
for sc in ['split_wc','split_dc']:
    r=evaluate(X.astype(np.float32),sc,tag='hand%d'%X.shape[1]); res[sc]=r
    print(json.dumps({k:r[k] for k in ('tag','split','n_query','macro_F1','ci_macro_F1','macro_AUC','ci_macro_AUC','ac_recall','ci_ac_recall')}))
json.dump(res,open(ROOT+'/smartscanner/code_icde/v2/artifacts/res_hand.json','w'),indent=2)
