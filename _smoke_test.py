# -*- encoding: utf-8 -*-
"""冒烟测试: 验证 Blackwell GB10 能跑模型前向+反向"""
import sys, time
sys.path.insert(0, '.')
import torch
from transformers import AutoTokenizer

# 把 rankcse/capsule 改成可选,避免缺包
sys.modules.setdefault('capsule_layer', __import__('types').ModuleType('capsule_layer'))
sys.modules['capsule_layer'].CapsuleLinear = None
sys.modules['capsule_layer'].CapsuleConv2d = None

from model import SimcseModel

DEVICE = torch.device('cuda')
MODEL = './SC_model_big_long_resnet_1d_24000'

print(f'[smoke] torch={torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}')

tok = AutoTokenizer.from_pretrained(MODEL)
print(f'[smoke] tokenizer loaded, vocab={tok.vocab_size}')

model = SimcseModel(pretrained_model=MODEL, pooling='last-avg', teacher_isavailable=False).to(DEVICE)
print(f'[smoke] model on {DEVICE}, n_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')

# maxlen=24000, batch=2, num_sen=3 -> shape (6, 24000)
maxlen = 24000
batch = 2
texts = ['pragma solidity ^0.8.0; contract A { uint x; }'] * (batch * 3)
enc = tok(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=maxlen)
input_ids = enc['input_ids'].to(DEVICE)
attention_mask = enc['attention_mask'].to(DEVICE)
print(f'[smoke] input_ids={tuple(input_ids.shape)}, dtype={input_ids.dtype}')

model.train()
torch.cuda.synchronize()
t0 = time.time()
ret = model(input_ids, attention_mask, None, None, None)
loss = ret[0]
loss.backward()
torch.cuda.synchronize()
dt = time.time() - t0
print(f'[smoke] fwd+bwd OK  loss={loss.item():.4f}  ret_count={len(ret)}  time={dt:.2f}s  peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB')
def _shape(x):
    return tuple(x.shape) if hasattr(x, 'shape') else type(x)
print(f'[smoke] ret[1]={_shape(ret[1])}  ret[2]={_shape(ret[2])}')
