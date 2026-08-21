# -*- encoding: utf-8 -*-
"""从 SASC parquet + acc_test_100_24000.json 构造 CCRNet 训练/评测数据
SimCSE 三元组: (origin, entailment, contradiction)
Test 三元组: (s1, s2, score)  score in [0,1]"""
import os, json, random, sys
import pandas as pd

RAW_DIR = '/home/qzqdz/Desktop/project/smart-contract-security/datasets/sasc/data/raw'
SPLITS  = '/home/qzqdz/Desktop/project/smart-contract-security/datasets/sasc/data/big-splits.csv'
ACC     = '/home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde/dataset/acc_test_100_24000.json'
OUT_DIR = '/home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde/dataset/sc_big_local'

os.makedirs(OUT_DIR, exist_ok=True)

def load_sasc_source_codes():
    """加载所有 parquet, 取出 contracts address → source_code 映射, 加 splits 信息"""
    splits_df = pd.read_csv(SPLITS)
    split_map = dict(zip(splits_df['contracts'], splits_df['split']))
    addr_to_code = {}
    for f in sorted(os.listdir(RAW_DIR)):
        if not f.endswith('.parquet'): continue
        path = os.path.join(RAW_DIR, f)
        df = pd.read_parquet(path, columns=['contracts', 'source_code'])
        for _, row in df.iterrows():
            addr_to_code[row['contracts']] = row['source_code']
        print(f'  loaded {f}: {len(df)} contracts, total={len(addr_to_code)}')
    return addr_to_code, split_map

def main():
    random.seed(42)
    addr_to_code, split_map = load_sasc_source_codes()
    by_split = {'train': [], 'val': [], 'test': []}
    for addr, code in addr_to_code.items():
        s = split_map.get(addr)
        if s in by_split:
            by_split[s].append((addr, code))
    for k, v in by_split.items():
        print(f'{k}: {len(v)} contracts')

    # 训练集 SimCSE 三元组 (origin, entailment, contradiction)
    # origin == entailment (同一条合约, dropout 起作用), contradiction 是另一条
    train_codes = [c for _, c in by_split['train']]
    random.shuffle(train_codes)
    print(f'Building SimCSE triplets from {len(train_codes)} train contracts...')
    triplets = []
    # 为节省 I/O, 只取 5000 条训练样本, 每条配 1 个随机 negative
    MAX_TRAIN = 5000
    for i in range(min(MAX_TRAIN, len(train_codes))):
        anchor = train_codes[i]
        # negative: 同一个 split 内不同位置 (确定性)
        neg = train_codes[(i + random.randint(1, len(train_codes)-1)) % len(train_codes)]
        triplets.append({'origin': anchor, 'entailment': anchor, 'contradiction': neg})
    train_path = os.path.join(OUT_DIR, 'train_triplets.json')
    with open(train_path, 'w') as f:
        json.dump(triplets, f)
    print(f'  wrote {train_path} ({len(triplets)} triplets)')

    # dev/test 用 (s1, s2, score) 形式: 同标签合约 score=1.0, 不同标签 score=0.0
    # val 做 dev
    def make_pairs(items, n_pairs=500):
        """生成 (s1, s2, score) pairs; 一半正样本(score=1.0=同一条),一半负样本(score=0.0=不同)"""
        out = []
        half = n_pairs // 2
        for _ in range(half):
            i = random.randrange(len(items))
            out.append([items[i][1], items[i][1], 1.0])  # 正样本:同一条
        for _ in range(n_pairs - half):
            a = random.randrange(len(items))
            b = random.randrange(len(items))
            while b == a:
                b = random.randrange(len(items))
            out.append([items[a][1], items[b][1], 0.0])  # 负样本:不同条
        return out

    dev_pairs = make_pairs(by_split['val'], 500)
    test_pairs = make_pairs(by_split['test'], 500)
    dev_path = os.path.join(OUT_DIR, 'dev_pairs.json')
    test_path = os.path.join(OUT_DIR, 'test_pairs.json')
    with open(dev_path, 'w') as f: json.dump(dev_pairs, f)
    with open(test_path, 'w') as f: json.dump(test_pairs, f)
    print(f'  wrote {dev_path} ({len(dev_pairs)} pairs)')
    print(f'  wrote {test_path} ({len(test_pairs)} pairs)')

    # acc_eval/knn_eval: 直接复用 acc_test_100_24000.json (TextDataset 格式)
    # 用 val 子集作为 acc_train, test 子集作为 acc_val
    acc_data = json.load(open(ACC))
    # 取 100 条 val + 100 条 test, 标签二元化(>=1 漏洞→1, else 0)
    acc_train = []
    acc_val = []
    pos = 0
    for item in acc_data:
        code = item['target_text']
        label = item.get('label', None)
        # label 字段类型多样, 二元化
        binary_label = 0 if (label is None or label == 'safe' or label == 0 or label == '0') else 1
        rec = {'target_text': code, 'label': binary_label}
        if pos < 200 and binary_label in (0, 1):
            acc_train.append(rec)
            pos += 1
        elif pos >= 200 and pos < 400 and binary_label in (0, 1):
            acc_val.append(rec)
            pos += 1
        if pos >= 400: break
    # 转 DataFrame JSON 期望格式: json with [{target_text,label}]
    at_path = os.path.join(OUT_DIR, 'acc_train.json')
    av_path = os.path.join(OUT_DIR, 'acc_val.json')
    with open(at_path, 'w') as f: json.dump(acc_train, f)
    with open(av_path, 'w') as f: json.dump(acc_val, f)
    print(f'  wrote {at_path} ({len(acc_train)} train)')
    print(f'  wrote {av_path} ({len(acc_val)} val)')
    print('DONE')

if __name__ == '__main__':
    main()
