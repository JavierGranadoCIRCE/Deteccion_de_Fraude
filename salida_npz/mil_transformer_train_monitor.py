#!/usr/bin/env python3
"""
mil_transformer_train_monitor.py
--------------------------------
Entrenador MIL (Transformer por ventana) con MONITORIZACIÓN integrada:
- Barra de progreso por batch (tqdm).
- Métricas por epoch (train_loss, val_loss, AUC, AP) impresas SIEMPRE.
- Early Stopping (patience, min_delta).
- ReduceLROnPlateau (scheduler).
- Mixed precision (autocast + GradScaler) para GPU.
- Logging a CSV (por epoch) y (opcional) a TensorBoard.

Uso típico:
  python -u mil_transformer_train_monitor.py \
    --data_dir salida_npz \
    --save_dir checkpoints \
    --epochs 15 \
    --batch_size 16 \
    --device cuda \
    --log_csv train_log.csv

Requisitos: torch, numpy, tqdm
Opcional: scikit-learn (para AUC/AP), tensorboard (si usas --log_tensorboard).
"""
import argparse, json, math, time, csv, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -------- Datos --------
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    meta = json.loads(str(d["meta"]))
    return X, meta

class BagsDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        if len(self.paths) == 0:
            raise RuntimeError("No .npz found in data_dir.")
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
    L = batch[0][0].shape[1]; C = batch[0][0].shape[2]
    B = len(batch)
    X = torch.zeros(B, maxW, L, C, dtype=torch.float32)
    mask = torch.zeros(B, maxW, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.float32)
    paths = []
    for i,(Xi,yi,pi) in enumerate(batch):
        W = Xi.shape[0]
        X[i,:W] = Xi; mask[i,:W] = True; y[i] = yi; paths.append(pi)
    return X, mask, y, paths

# -------- Modelo --------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class WindowEncoder(nn.Module):
    def __init__(self, d_model=128, depth=2, heads=4, dropout=0.1, in_ch=2):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4*d_model,
                                               dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pos = PositionalEncoding(d_model)
    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        return h.mean(dim=1)  # (B*W, d_model)



class AttentionMIL(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.U = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, 1, bias=False)

    def forward(self, H, mask):
        A = torch.tanh(self.U(H))
        A = self.V(A).squeeze(-1)           # (B, W)
        # >>> CAMBIA ESTA LÍNEA <<<
        A = A.masked_fill(~mask, -1e4)      # antes: -1e9 (overflow en fp16)
        A = torch.softmax(A, dim=1)
        bag = torch.bmm(A.unsqueeze(1), H).squeeze(1)
        return bag, A


class MILModel(nn.Module):
    def __init__(self, d_model=128, depth=2, heads=4, dropout=0.1, in_ch=2):
        super().__init__()
        self.enc = WindowEncoder(d_model, depth, heads, dropout, in_ch)
        self.mil = AttentionMIL(d_model)
        self.cls = nn.Linear(d_model, 1)
    def forward(self, X, mask):
        B, W, L, C = X.shape
        H = self.enc(X.reshape(B*W, L, C)).view(B, W, -1)
        bag, A = self.mil(H, mask)
        logit = self.cls(bag).squeeze(-1)
        return logit, A

# -------- Entrenamiento --------
def train_epoch(model, loader, opt, device, scaler=None):
    model.train()
    total, n = 0.0, 0
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Train", leave=False)
    for X, mask, y, _ in iterator:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logit, _ = model(X, mask)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logit, _ = model(X, mask)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            opt.step()
        bs = X.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    ys, ps = [], []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Val", leave=False)
    for X, mask, y, _ in iterator:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        logit, _ = model(X, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        bs = X.size(0)
        total += loss.item() * bs
        n += bs
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.cpu().numpy())
    ys = np.concatenate(ys) if ys else np.array([])
    ps = np.concatenate(ps) if ps else np.array([])
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
        ap = average_precision_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
    except Exception:
        auc, ap = float('nan'), float('nan')
    return total / max(1, n), auc, ap

def maybe_tensorboard(log_tensorboard, save_dir):
    if not log_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(Path(save_dir) / "tboard"))
    except Exception:
        print("⚠️ TensorBoard no disponible. Instalación recomendada: pip install tensorboard")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--early_patience", type=int, default=4)
    ap.add_argument("--min_delta", type=float, default=0.002, help="Mejora mínima en val_loss para resetear patience")
    ap.add_argument("--log_csv", default=None, help="Ruta CSV para registrar métricas por epoch")
    ap.add_argument("--log_tensorboard", action="store_true")
    ap.add_argument("--mixed_precision", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.npz"))
    if len(paths) < 2:
        raise RuntimeError(f"Se necesitan ≥2 .npz en {data_dir}.")
    n = len(paths); n_val = max(1, int(n * args.val_split))
    val_paths = paths[:n_val]; tr_paths = paths[n_val:]

    train_ds = BagsDataset(tr_paths)
    val_ds = BagsDataset(val_paths)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_bags, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bags, num_workers=0)

    device = args.device
    model = MILModel(d_model=args.d_model, depth=args.depth, heads=args.heads, dropout=args.dropout, in_ch=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and (device == "cuda"))
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf'); best_path = save_dir / "best.pt"
    patience_left = args.early_patience

    # CSV logging setup
    csv_writer = None; csv_fh = None
    if args.log_csv:
        csv_fh = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "AUC", "AP", "lr"])

    tb = maybe_tensorboard(args.log_tensorboard, save_dir)

    print(f"🧪 Iniciando entrenamiento con {len(tr_paths)} train y {len(val_paths)} val bolsas.")
    print(f"Dispositivo: {device}  |  Mixed precision: {bool(scaler and scaler.is_enabled())}")
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_ld, opt, device, scaler)
        val_loss, auc, ap = eval_epoch(model, val_ld, device)
        lr_now = opt.param_groups[0]['lr']
        dt = time.time() - t0

        # Scheduler
        scheduler.step(val_loss)

        # Logging
        print(f"[Epoch {ep:02d}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  AUC={auc:.4f}  AP={ap:.4f}  lr={lr_now:.2e}  ({dt:.1f}s)")
        if csv_writer:
            csv_writer.writerow([ep, f"{tr_loss:.6f}", f"{val_loss:.6f}", f"{auc:.6f}", f"{ap:.6f}", f"{lr_now:.6e}"])
            csv_fh.flush()
        if tb:
            tb.add_scalar("Loss/train", tr_loss, ep)
            tb.add_scalar("Loss/val", val_loss, ep)
            if not (math.isnan(auc) or math.isinf(auc)):
                tb.add_scalar("AUC/val", auc, ep)
            if not (math.isnan(ap) or math.isinf(ap)):
                tb.add_scalar("AP/val", ap, ep)
            tb.add_scalar("LR", lr_now, ep)

        # Early stopping
        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> guardado {best_path}")
            patience_left = args.early_patience
        else:
            patience_left -= 1
            print(f"  -> sin mejora (best={best_val:.4f}), paciencia restante: {patience_left}")
            if patience_left <= 0:
                print("⏹️ Early stopping activado.")
                break

    if csv_fh: csv_fh.close()
    if tb: tb.close()
    print(f"[OK] Fin. Mejor val_loss={best_val:.4f}  |  Modelo: {best_path}")
if __name__ == "__main__":
    main()
