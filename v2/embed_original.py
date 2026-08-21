# -*- coding: utf-8 -*-
"""Embed the 16k benchmark with the ORIGINAL CCRNet checkpoint (emb_dim=3)."""
import sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/qzqdz/Desktop/project/smart-contract-security/smartscanner/code_icde')
import model as M
M.ListNet=lambda *a,**k: nn.Identity()      # guard the None-crash (teacher unused)
from model import SimcseModel
ROOT='/home/qzqdz/Desktop/project/smart-contract-security'; CI=ROOT+'/smartscanner/code_icde'
BEN=CI+'/v2/bench'
tok=np.load(BEN+'/tokens.npy')
m=SimcseModel(CI+'/SC_model_big_long_resnet_1d_24000', pooling='last-avg', only_embeddings=True)
sd=torch.load(CI+'/saved_model/ccrnet_gb10/pytorch_model.bin', map_location='cpu')
missing,unexpected=m.load_state_dict(sd, strict=False)
print('loaded ckpt; missing(sample)',list(missing)[:4],'unexpected(sample)',list(unexpected)[:4])
m.eval()
emb=np.zeros((len(tok),512),dtype=np.float32); B=64
with torch.no_grad():
    for s in range(0,len(tok),B):
        x=torch.from_numpy(tok[s:s+B].astype(np.int64))
        rep=m.encode(x) if hasattr(m,'encode') else m.layers(m.embedding(x).permute(0,2,1))
        emb[s:s+B]=rep.numpy()
        if s % (B*40)==0: print('embed',s,flush=True)
np.save(BEN+'/emb_original.npy',emb)
print('emb_original saved',emb.shape)
