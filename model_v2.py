# -*- coding: utf-8 -*-
"""CCRNet-v2: configurable CC-channel (emb_dim), no teacher, clean init.
Reuses exact building blocks from model.py (CompositionalEmbedding/ResBlock1D/
Similarity/Divergence/simcse_sup_loss). Submodule names 'embedding'/'layers'
match the original SimcseModel state_dict so emb_dim=3 can load the checkpoint.
"""
import torch, torch.nn as nn
from model import (CompositionalEmbedding, ResBlock1D, Similarity, Divergence,
                   simcse_sup_loss)

def _stack(emb_dim):
    return nn.Sequential(
        nn.Conv1d(emb_dim,128,3,stride=3), nn.BatchNorm1d(128), nn.LeakyReLU(),
        ResBlock1D(128,128), nn.MaxPool1d(3,3),
        ResBlock1D(128,128), nn.MaxPool1d(3,3),
        nn.Conv1d(128,256,3,1,padding=1), nn.BatchNorm1d(256), nn.LeakyReLU(),
        ResBlock1D(256,256), nn.MaxPool1d(3,3),
        ResBlock1D(256,256), nn.MaxPool1d(3,3),
        ResBlock1D(256,256), nn.MaxPool1d(3,3),
        nn.Conv1d(256,512,3,1,padding=1), nn.BatchNorm1d(512), nn.LeakyReLU(),
        ResBlock1D(512,512), nn.MaxPool1d(3,3),
        nn.Conv1d(512,512,1,1), nn.BatchNorm1d(512), nn.LeakyReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Dropout(0.4), nn.Flatten())

class CCRNetV2(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, num_codebook=8):
        super().__init__()
        self.num_sen=3
        self.embedding=CompositionalEmbedding(vocab_size, emb_dim, num_codebook, None, weighted=True)
        self.layers=_stack(emb_dim)
        self.sim=Similarity(temp=0.05); self.div=Divergence(beta_=0.5)
    def encode(self, input_ids):
        e=self.embedding(input_ids).permute(0,2,1)
        return self.layers(e)
    def forward(self, input_ids, use_contrastive=True):
        """input_ids: (3B, L) flattened triples. Returns (loss, rep)."""
        rep=self.encode(input_ids)
        if not use_contrastive:
            # degenerate control: still returns a loss so the loop runs, but no InfoNCE
            return rep.pow(2).mean()*0.0+torch.tensor(0.0,device=rep.device,requires_grad=True), rep
        loss=simcse_sup_loss(rep)
        B=input_ids.size(0)//3
        r3=rep.view(B,3,-1); z1,z2,z3=r3[:,0],r3[:,1],r3[:,2]
        a=self.sim(z1.unsqueeze(1),z2.unsqueeze(0)); b=self.sim(z2.unsqueeze(1),z1.unsqueeze(0))
        a=torch.cat([a,self.sim(z1.unsqueeze(1),z3.unsqueeze(0))],1)
        b=torch.cat([b,self.sim(z2.unsqueeze(1),z3.unsqueeze(0))],1)
        sd=self.div(a.softmax(-1).clamp(min=1e-7), b.softmax(-1).clamp(min=1e-7))
        return loss+0.1*sd, rep
