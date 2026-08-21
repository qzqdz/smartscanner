# -*- coding: utf-8 -*-
"""exp15 clean-label relabel, scoped to the 3600 in3600 benchmark contracts.
Self-contained (no Windows-path common.py). Reuses the exact s2 PROMPT + qwen call.
Resumable JSONL cache (append-only, dedup-by-addr, schema_version isolation).
Run:  python relabel_3600.py --workers 8   (re-run same cmd to resume after a break)
"""
import os, re, json, time, glob, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT   = '/home/qzqdz/Desktop/project/smart-contract-security'
SASC   = ROOT + '/datasets/sasc'
HERE   = os.path.dirname(os.path.abspath(__file__))
MANI   = ROOT + '/smartscanner/code_icde/v2/bench/manifest_3600.json'
CACHE  = HERE + '/clean_labels.jsonl'
API    = 'http://192.168.10.1:8000/v1/chat/completions'
MODEL  = 'qwen3.8-27b'; APIKEY = 'xx'
SCHEMA_VER = 'v2'
CODE_CAP, HEAD, TAIL = 120000, 90000, 30000
CLASSES = ['access_control', 'reentrancy', 'arithmetic', 'unchecked_calls']

PROMPT = """You are a senior smart-contract security auditor. The Solidity source below has ABSOLUTE line numbers prefixed as `  NNN| `. Audit for these 4 vulnerability classes:
- access_control: missing/broken authorization on state-changing or privileged funcs (owner, mint, withdraw, selfdestruct) with NO effective guard.
- reentrancy: external call before state update with NO reentrancy guard / checks-effects-interactions.
- arithmetic: integer overflow/underflow or unsafe math with NO SafeMath / unchecked-block misuse / >=0.8 protection absent.
- unchecked_calls: low-level call/send/transfer/delegatecall whose return value is ignored.

For EACH class you CONFIRM as a live issue, emit a finding with the REAL line number you see. For EACH class you judge SAFE because a defense is present, emit a safe_receipt naming the exact mitigation and its line (e.g. `nonReentrant` modifier @ line 412, `onlyOwner` @ 88, `SafeMath.add` @ 205, `require(success)` @ 130). If a class simply does not apply (no such code path), omit it from both.

Reply ONLY compact JSON, no prose:
{"findings":[{"class":"reentrancy","function":"withdraw","line":412,"snippet":"(bool s,)=msg.sender.call{value:amt}(\\"\\");","defense":"none","verdict":"live"}],
 "safe_receipts":[{"class":"arithmetic","defense":"SafeMath.sub","line":205,"evidence":"all subtraction via SafeMath"}],
 "summary":"<one sentence overall>"}
verdict is "live" (exploitable path, no mitigation) or "mitigated" (issue shape present but guarded)."""

def norm(a):        # strip 0x, lowercase hex
    a = str(a).lower()
    return a[2:] if a.startswith('0x') else a

def number_lines(code):
    return '\n'.join(f'{i+1:>5}| {ln}' for i, ln in enumerate(code.split('\n')))

def clip_numbered(code):
    nl = number_lines(code)
    if len(nl) <= CODE_CAP: return nl
    return nl[:HEAD] + '\n     |  ... [OMITTED MIDDLE] ...\n' + nl[-TAIL:]

def extract_json(txt):
    if not txt: return None
    m = re.search(r'\{.*\}', txt, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def chat(prompt, code, max_tokens=700, retries=3):
    body = json.dumps({'model': MODEL,
        'messages':[{'role':'user','content':prompt+'\n\n<source>\n'+code+'\n</source>'}],
        'temperature':0.0,'max_tokens':max_tokens,
        'chat_template_kwargs':{'enable_thinking':False}}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API, data=body,
                headers={'Content-Type':'application/json','Authorization':'Bearer '+APIKEY})
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.loads(r.read())
            return j['choices'][0]['message']['content']
        except Exception:
            if attempt == retries-1: return None
            time.sleep(2*(attempt+1))
    return None

def label_from(findings):
    lab = {c: False for c in CLASSES}
    for f in findings or []:
        c = f.get('class','').replace('-','_')
        if c in lab and f.get('verdict') == 'live':
            lab[c] = True
    lab['safe'] = not any(lab[c] for c in CLASSES)
    return lab

def load_cache(path):
    d={}
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line=line.strip()
            if not line: continue
            try: o=json.loads(line); d[o['addr']]=o
            except Exception: pass
    return d

def is_stale(e):
    return (not e) or e.get('schema_version') != SCHEMA_VER

def code_map(want):
    """one pass over parquets -> {norm_addr: source_code} for wanted addrs."""
    import pandas as pd
    out={}
    for f in sorted(glob.glob(os.path.join(SASC,'data/raw/*.parquet'))):
        df = pd.read_parquet(f, columns=['contracts','source_code'])
        for a,src in zip(df['contracts'], df['source_code']):
            na=norm(a)
            if na in want and na not in out:
                out[na]=src or ''
        if len(out)>=len(want): break
    return out

def work(addr, code):
    try:
        j = extract_json(chat(PROMPT, clip_numbered(code)))
        if j is None: return None
        findings=[f for f in (j.get('findings') or []) if isinstance(f,dict)]
        receipts=[r for r in (j.get('safe_receipts') or []) if isinstance(r,dict)]
    except Exception:
        return None
    return {'addr':addr,'schema_version':SCHEMA_VER,
            'labels':label_from(findings),'findings':findings,'safe_receipts':receipts,
            'mitigated_classes':sorted({r.get('class','').replace('-','_') for r in receipts if r.get('class')} & set(CLASSES)),
            'summary':(j.get('summary') or '')[:300]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--limit',type=int,default=None)
    a=ap.parse_args()
    want=set(norm(x) for x in json.load(open(MANI))['addrs'])
    cache=load_cache(CACHE)
    todo=[x for x in want if is_stale(cache.get(x))]
    if a.limit: todo=todo[:a.limit]
    print(f'[relabel] want={len(want)} cached={len(cache)} todo={len(todo)}',flush=True)
    if not todo:
        print('[relabel] nothing to do — already complete',flush=True); return
    cm=code_map(set(todo))
    print(f'[relabel] loaded source for {len(cm)}/{len(todo)} addrs',flush=True)
    done=0; consec=0; ABORT=50; t0=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(work, x, cm.get(x,'')): x for x in todo}
        for fut in as_completed(futs):
            try: r=fut.result()
            except Exception: r=None
            if r:
                # append under lock-free single-writer (as_completed serial) 
                open(CACHE,'a',encoding='utf-8').write(json.dumps(r,ensure_ascii=False)+'\n')
                done+=1; consec=0
            else:
                consec+=1
                if consec>=ABORT:
                    print(f'[relabel] ABORT: {ABORT} consecutive failures — vLLM/tunnel down. '
                          f'saved {done}, cache is resumable: rerun same cmd.',flush=True)
                    return
            if done and done%50==0:
                rate=done/(time.time()-t0); eta=(len(todo)-done)/rate/3600 if rate else 0
                print(f'[relabel] {done}/{len(todo)}  {rate:.2f}/s  ETA {eta:.1f}h',flush=True)
    print(f'[relabel] DONE +{done}, failed={len(todo)-done}, cache now ~{len(cache)+done}',flush=True)

if __name__=='__main__':
    main()
