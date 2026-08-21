# -*- coding: utf-8 -*-
"""C2 定位评测口径(方法无关的尺子). 单位=函数(花括号配平, 带行号).
Hit@k(类c) = 某合约按类c打分排序的 top-k 函数, 其行区间是否命中该合约任一 GT 漏洞行.
分母 = 含该类且GT行落在某函数内的合约; GT行在所有函数外的单独计数(函数级不可命中).
基线随时可插: 传 scorer(func_texts, cls_idx, contract_text)->np.array(每函数分)."""
import re, os, json, glob, numpy as np
from math import comb
SB=os.path.join(os.path.dirname(__file__),'../../..','data/smartbugs-curated')
SB=os.path.abspath(SB)
# SmartBugs 类 -> SASC 4 类 idx (y_ac,y_re,y_ar,y_uc)
# arithmetic 语义不符(SASC y_ar=Slither divide-before-multiply; SmartBugs=溢出/下溢),仅诊断不作头条
CMAP={'access_control':0,'reentrancy':1,'arithmetic':2,'unchecked_low_level_calls':3}
CLS=['access_control','reentrancy','arithmetic','unchecked_low_level_calls']
HEAD=re.compile(r'\b(function|constructor|modifier|receive|fallback)\b[^;{}]*\{')
def strip_comments_keep_lines(code):
    """剥注释但保留所有换行(GT行号仍有效). 防注释里的<yes>/@vulnerable_at_lines答案泄漏 + 防注释mint假函数."""
    out=[]; i=0; n=len(code)
    while i<n:
        c=code[i]; nx=code[i+1] if i+1<n else ''
        if c=='/' and nx=='/':
            while i<n and code[i]!='\n': out.append(' '); i+=1
        elif c=='/' and nx=='*':
            out.append('  '); i+=2
            while i<n and not (code[i]=='*' and i+1<n and code[i+1]=='/'):
                out.append('\n' if code[i]=='\n' else ' '); i+=1
            if i<n: out.append('  '); i+=2
        elif c=='"':
            out.append('"'); i+=1
            while i<n and code[i]!='"':
                if code[i]=='\\' and i+1<n: out.append('  '); i+=2; continue
                out.append('\n' if code[i]=='\n' else ' '); i+=1
            if i<n: out.append('"'); i+=1
        elif c=="'":
            out.append("'"); i+=1
            while i<n and code[i]!="'":
                if code[i]=='\\' and i+1<n: out.append('  '); i+=2; continue
                out.append('\n' if code[i]=='\n' else ' '); i+=1
            if i<n: out.append("'"); i+=1
        else: out.append(c); i+=1
    return ''.join(out)
def parse_funcs(code):
    """返回 [(start_line,end_line,text)] 逐函数. code 必须已剥注释(保行号)."""
    spans=[]; i=0; n=len(code)
    for m in HEAD.finditer(code):
        b=m.end()-1  # 位置在 '{'
        depth=0; j=b
        while j<n:
            if code[j]=='{':depth+=1
            elif code[j]=='}':
                depth-=1
                if depth==0: break
            j+=1
        s=m.start(); e=min(j+1,n)
        sl=code.count('\n',0,s)+1; el=code.count('\n',0,e)+1
        spans.append((sl,el,code[s:e]))
    return spans
def load_gt():
    v=json.load(open(os.path.join(SB,'vulnerabilities.json')))
    items=[]  # {path,code,funcs,by_cls:{cidx:set(lines)}}
    for e in v:
        rel=e['path']; cats={}
        for vu in e['vulnerabilities']:
            c=vu['category']
            if c in CMAP:
                cats.setdefault(CMAP[c],set()).update(vu.get('lines',[]))
        if not cats: continue
        fp=os.path.join(SB,rel)
        if not os.path.exists(fp): continue
        raw=open(fp,encoding='utf-8',errors='ignore').read()
        code=strip_comments_keep_lines(raw)   # 剥注释(保行号)防答案泄漏
        items.append({'path':rel,'code':code,'funcs':parse_funcs(code),'by_cls':cats})
    return items
def line_in_func(funcs, line):
    for idx,(sl,el,_) in enumerate(funcs):
        if sl<=line<=el: return idx
    return -1
def pos_funcs(funcs, gt_lines):
    """GT行落在哪些函数内 -> 正例函数下标集合."""
    s=set()
    for L in gt_lines:
        fi=line_in_func(funcs,L)
        if fi>=0: s.add(fi)
    return s
def rand_hit_expect(F,P,k):
    """解析期望: top-k 至少命中一个正例函数 = 1 - C(F-P,k)/C(F,k)."""
    k=min(k,F)
    if P<=0 or F<=0: return None
    if F-P<k: return 1.0
    return 1.0 - comb(F-P,k)/comb(F,k)
def rand_mrr_expect(F,P):
    """随机排序下 首个正例的 1/rank 期望 = sum_{r} P(first hit at r)/r; 近似用 (P+1)/(F+1) 为首命中位置期望的倒数上界.
    这里用精确: 首正例在位置r的概率 = C(F-P, r-1)/C(F,r-1) * P/(F-r+1)."""
    if P<=0 or F<=0: return None
    tot=0.0
    for r in range(1,F-P+2):
        # P(all first r-1 are negatives) * P(r-th is positive)
        num=1.0
        for j in range(r-1): num*= (F-P-j)/(F-j)
        p_first=num*(P/(F-(r-1)))
        tot+= p_first*(1.0/r)
    return tot
def evaluate(scorer, k_list=(1,3), B=1000, seed=42):
    """scorer(func_texts, cidx, contract_code)->每函数分. 报 Hit@k + MRR(rank归一) + 解析random期望."""
    items=load_gt()
    per={c:{**{f'hit@{k}':[] for k in k_list},'rr':[],'F':[],'P':[]} for c in range(4)}
    excluded={c:0 for c in range(4)}
    for it in items:
        funcs=it['funcs']
        if not funcs: continue
        ftexts=[f[2] for f in funcs]
        for cidx,lines in it['by_cls'].items():
            pf=pos_funcs(funcs,lines)
            if not pf: excluded[cidx]+=1; continue
            F=len(funcs); P=len(pf)
            sc=scorer(ftexts,cidx,it['code'])
            order=np.argsort(-sc)  # 高分在前(稳定,平分按原序)
            for k in k_list:
                per[cidx][f'hit@{k}'].append(1 if (set(order[:k].tolist())&pf) else 0)
            # reciprocal rank of first hit
            rr=0.0
            for rank,fi in enumerate(order.tolist(),1):
                if fi in pf: rr=1.0/rank; break
            per[cidx]['rr'].append(rr); per[cidx]['F'].append(F); per[cidx]['P'].append(P)
    rng=np.random.RandomState(seed); out={}
    for c in range(4):
        o={'excluded_gt_outside_func':excluded[c]}
        Fs=per[c]['F']; Ps=per[c]['P']
        if Fs:
            o['n']=len(Fs); o['n_funcs_avg']=round(float(np.mean(Fs)),2); o['pos_funcs_avg']=round(float(np.mean(Ps)),2)
            for k in k_list:
                a=np.array(per[c][f'hit@{k}'],float)
                bs=[rng.choice(a,len(a),replace=True).mean() for _ in range(B)]
                rexp=float(np.mean([rand_hit_expect(F,P,k) for F,P in zip(Fs,Ps)]))
                o[f'hit@{k}']={'mean':round(float(a.mean()),4),
                    'ci':[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)],
                    'random_expect':round(rexp,4)}
            a=np.array(per[c]['rr'],float)
            bs=[rng.choice(a,len(a),replace=True).mean() for _ in range(B)]
            rexp=float(np.mean([rand_mrr_expect(F,P) for F,P in zip(Fs,Ps)]))
            o['MRR']={'mean':round(float(a.mean()),4),
                'ci':[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)],
                'random_expect':round(rexp,4)}
        out[CLS[c]]=o
    return out

# ---------- 廉价基线(验尺子) ----------
SINK={0:re.compile(r'selfdestruct|suicide|delegatecall|tx\.origin|\bowner\s*=|onlyOwner|\bmint\b'),
      1:re.compile(r'\.call\s*\{\s*value|\.call\.value|\.call\s*\(|\.send\s*\(|\.transfer\s*\('),
      2:re.compile(r'[+\-*]|SafeMath|\bunchecked\b'),
      3:re.compile(r'\.send\s*\(|\.call\s*[\({]|\.delegatecall')}
def sc_random(ft,c,code):
    rng=np.random.RandomState(abs(hash((len(code),c)))%(2**31)); return rng.rand(len(ft))
def sc_regex(ft,c,code):
    rx=SINK[c]; return np.array([len(rx.findall(t)) for t in ft],float)
if __name__=='__main__':
    r=evaluate(sc_regex)
    print("=== regex-sink (剥注释后) vs 解析random期望 ===")
    for c in CLS:
        d=r[c]
        if 'n' not in d: print(f"  {c}: NA"); continue
        h1=d['hit@1']; h3=d['hit@3']; mr=d['MRR']
        print(f"  {c:26s} n={d['n']:2d} F̄={d['n_funcs_avg']:4.1f} P̄={d['pos_funcs_avg']:.1f} | "
              f"Hit@1={h1['mean']:.3f}(rand {h1['random_expect']:.3f}) ci{h1['ci']} | "
              f"Hit@3={h3['mean']:.3f}(rand {h3['random_expect']:.3f}) | "
              f"MRR={mr['mean']:.3f}(rand {mr['random_expect']:.3f})")
