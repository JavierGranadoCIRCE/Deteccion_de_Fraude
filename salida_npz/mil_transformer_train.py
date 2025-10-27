#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]; meta = json.loads(str(d["meta"]))
    return X, meta
class BagsDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.labels = []
        for p in self.paths:
            _, meta = load_npz(p)
            self.labels.append(int(meta.get("label", 0)))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        X, meta = load_npz(self.paths[i])
        y = int(meta.get("label", 0))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), str(self.paths[i])
def collate_bags(batch):
    maxW = max(x[0].shape[0] for x in batch)
    L = batch[0][0].shape[1]; C = batch[0][0].shape[2]; B = len(batch)
    X = torch.zeros(B, maxW, L, C); mask = torch.zeros(B, maxW, dtype=torch.bool)
    y = torch.zeros(B); paths = []
    for i,(Xi,yi,pi) in enumerate(batch):
        W = Xi.shape[0]; X[i,:W] = Xi; mask[i,:W] = True; y[i] = yi; paths.append(pi)
    return X, mask, y, paths
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
class WindowEncoder(nn.Module):
    def __init__(self, d=128, depth=2, heads=4, drop=0.1, in_ch=2):
        super().__init__()
        self.proj = nn.Linear(in_ch, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4*d, dropout=drop, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pos = PositionalEncoding(d)
    def forward(self, x):
        h = self.proj(x); h = self.pos(h); h = self.enc(h); return h.mean(dim=1)
class AttentionMIL(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.U = nn.Linear(d, d); self.V = nn.Linear(d, 1, bias=False)
    def forward(self, H, mask):
        A = torch.tanh(self.U(H)); A = self.V(A).squeeze(-1)
        A = A.masked_fill(~mask, -1e9); A = torch.softmax(A, dim=1)
        bag = torch.bmm(A.unsqueeze(1), H).squeeze(1)
        return bag, A
class Model(nn.Module):
    def __init__(self, d=128, depth=2, heads=4, drop=0.1, in_ch=2):
        super().__init__()
        self.win = WindowEncoder(d, depth, heads, drop, in_ch)
        self.mil = AttentionMIL(d)
        self.cls = nn.Linear(d,1)
    def forward(self, X, mask):
        B,W,L,C = X.shape
        H = self.win(X.reshape(B*W,L,C)).view(B,W,-1)
        bag, A = self.mil(H, mask)
        logit = self.cls(bag).squeeze(-1); return logit, A
def train_epoch(m, ld, opt, dev):
    m.train(); tot=0.0
    for X,mask,y,_ in ld:
        X,mask,y = X.to(dev),mask.to(dev),y.to(dev)
        opt.zero_grad(); logit,_=m(X,mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit,y)
        loss.backward(); opt.step(); tot += loss.item()*X.size(0)
    return tot/len(ld.dataset)
@torch.no_grad()
def eval_epoch(m, ld, dev):
    m.eval(); tot=0.0; ys=[]; ps=[]
    for X,mask,y,_ in ld:
        X,mask,y = X.to(dev),mask.to(dev),y.to(dev)
        logit,_=m(X,mask); loss = torch.nn.functional.binary_cross_entropy_with_logits(logit,y)
        tot += loss.item()*X.size(0); prob=torch.sigmoid(logit)
        ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
    import numpy as np
    ys=np.concatenate(ys); ps=np.concatenate(ps)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc=roc_auc_score(ys,ps); ap=average_precision_score(ys,ps)
    except Exception: auc=float('nan'); ap=float('nan')
    return tot/len(ld.dataset), auc, ap
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val_split", type=float, default=0.1)
    args = ap.parse_args()
    paths = sorted(Path(args.data_dir).glob("*.npz"))
    if len(paths)<2: raise RuntimeError("Se necesitan ≥2 .npz.")
    n=len(paths); n_val=max(1,int(n*args.val_split)); val=paths[:n_val]; tr=paths[n_val:]
    trds=BagsDataset(tr); vlds=BagsDataset(val)
    trld=DataLoader(trds,batch_size=args.batch_size,shuffle=True,collate_fn=collate_bags)
    vlld=DataLoader(vlds,batch_size=args.batch_size,shuffle=False,collate_fn=collate_bags)
    dev=args.device; m=Model(args.d_model,args.depth,args.heads,args.dropout,2).to(dev)
    opt=torch.optim.AdamW(m.parameters(), lr=args.lr)
    best=float('inf'); out=Path(args.save_dir); out.mkdir(parents=True, exist_ok=True); bestp=out/"best.pt"
    for ep in range(1,args.epochs+1):
        trl=train_epoch(m,trld,opt,dev); vall,auc,ap=eval_epoch(m,vlld,dev)
        print(f"[Epoch {ep}] train_loss={trl:.4f} val_loss={vall:.4f} AUC={auc:.4f} AP={ap:.4f}")
        if vall<best: best=vall; torch.save(m.state_dict(), bestp); print(f"  -> guardado {bestp}")
    print("[OK] Fin. Mejor val_loss=",best)
if __name__=="__main__": main()
