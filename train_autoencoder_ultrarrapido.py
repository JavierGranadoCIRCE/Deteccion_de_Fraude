# ------------------------------------------------------------
# train_autoencoder_ultrarrapido.py
# Autoencoder Conv1D + GRU (30× más rápido que Transformer)
# ------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from comet_ml import Experiment


# ============================================================
#   MODELO ULTRARRÁPIDO
# ============================================================
class UltraFastAutoencoder(nn.Module):
    def __init__(self, feat_dim=2, hidden=64):
        super().__init__()

        # --- Encoder ---
        self.conv1 = nn.Conv1d(feat_dim, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

        self.gru_enc = nn.GRU(64, hidden, batch_first=True)

        # --- Decoder ---
        self.gru_dec = nn.GRU(hidden, 64, batch_first=True)
        self.deconv1 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.deconv2 = nn.Conv1d(32, feat_dim, kernel_size=5, padding=2)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)                 # (B, C, T)

        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.permute(0, 2, 1)                 # (B, T, 64)

        _, h_last = self.gru_enc(h)           # (1, B, hidden)

        dec_input = h_last.repeat(1, h.shape[1], 1).permute(1, 0, 2)

        out, _ = self.gru_dec(dec_input)      # (B, T, 64)
        out = out.permute(0, 2, 1)

        out = self.relu(self.deconv1(out))
        out = self.deconv2(out)

        return out.permute(0, 2, 1)           # (B, T, feat_dim)


# ============================================================
#   DATASET STREAMING
# ============================================================
class VentanasDataset(Dataset):
    def __init__(self, root, ventana, stride):
        self.root = root
        self.ventana = ventana
        self.stride = stride
        self.files = sorted(os.listdir(root))

        self.index = []
        for f in tqdm(self.files, desc="Escaneando"):
            d = np.load(os.path.join(root, f))
            L = d["energia"].shape[0]
            n = (L - ventana) // stride + 1
            self.index.append(n)

        self.total = sum(self.index)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        cum = 0
        for fi, n in enumerate(self.index):
            if idx < cum + n:
                break
            cum += n

        local = idx - cum
        file = self.files[fi]

        d = np.load(os.path.join(self.root, file))
        ini = local * self.stride
        fin = ini + self.ventana

        x = np.stack([
            d["energia"][ini:fin],
            d["fallo"][ini:fin]
        ], axis=1).astype(np.float32)

        # Normalización
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)

        return torch.tensor(x)


# ============================================================
#   ENTRENAMIENTO
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = Experiment(
        api_key=args.api_key,
        project_name="fraude-autoencoder-ultrarrapido",
        workspace="javier-granado"
    )

    ds = VentanasDataset(args.data_dir, args.ventana, args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"[INFO] Total ventanas: {len(ds):,}")

    model = UltraFastAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            batch = batch.to(device)

            pred = model(batch)
            loss = loss_fn(pred, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": np.mean(losses)})

        experiment.log_metric("loss", np.mean(losses), epoch=epoch)
        torch.save(model.state_dict(), f"autoencoder_ultra_ep{epoch}.pth")

    print("[FIN] Entrenamiento completado.")


# ============================================================
#   MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ventana", type=int, default=168)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--api_key", type=str, required=True)
    args = p.parse_args()
    train(args)
