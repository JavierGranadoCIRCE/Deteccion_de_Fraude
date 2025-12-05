import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
#  AUTOENCODER TRANSFORMER GLOBAL (entrada completa 17520 x 2)
# ============================================================
class GlobalTransformerAutoencoder(nn.Module):
    def __init__(self, dim=128, heads=8, layers=4, seq_len=17520):
        super().__init__()

        self.seq_len = seq_len
        self.in_proj = nn.Linear(2, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)

        self.out_proj = nn.Linear(dim, 2)

    def forward(self, x):
        h = self.in_proj(x)
        z = self.encoder(h)
        out = self.decoder(z, z)
        return self.out_proj(out)


# ============================================================
# DATASET global (17520 x 2)
# ============================================================
class ContadorDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        energia = d["energia"].astype(np.float32)
        fallo = d["fallo"].astype(np.float32)

        x = np.stack([energia, fallo], axis=1)  # (17520, 2)
        return torch.tensor(x)


# ============================================================
# FUNCIÓN PARA CALCULAR ERROR POR ARCHIVO (L2 reconstruction)
# ============================================================
def reconstruction_error(model, file, device):
    d = np.load(file)
    energia = d["energia"].astype(np.float32)
    fallo = d["fallo"].astype(np.float32)
    x = np.stack([energia, fallo], axis=1)
    x = torch.tensor(x).unsqueeze(0).to(device)  # (1,17520,2)

    with torch.no_grad():
        out = model(x)
    err = torch.mean((out - x)**2).item()
    return err


# ============================================================
# MAIN
# ============================================================
def main():

    path_no_fraude = "./aplanados_v5/dataset_no_fraude"
    path_fraude    = "./aplanados_v5/dataset_fraude"

    files_nof = [os.path.join(path_no_fraude, f) for f in os.listdir(path_no_fraude) if f.endswith(".npz")]
    files_f   = [os.path.join(path_fraude, f) for f in os.listdir(path_fraude) if f.endswith(".npz")]

    print(f"[INFO] Contadores NO FRAUDE: {len(files_nof)}")
    print(f"[INFO] Contadores FRAUDE   : {len(files_f)}")

    # --------------------------------------------------------
    # Train/Val split solo para NO FRAUDE (unsupervised)
    # --------------------------------------------------------
    train_files, val_nof_files = train_test_split(files_nof, test_size=0.2, random_state=42)

    train_set = ContadorDataset(train_files)
    val_set   = ContadorDataset(val_nof_files)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------
    model = GlobalTransformerAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    EPOCHS = 20

    print("[INFO] Entrenando modelo GLOBAL...")
    for ep in range(EPOCHS):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"[Epoch {ep+1}/{EPOCHS}] Loss: {np.mean(losses):.6f}")

    # Guardado
    torch.save(model.state_dict(), "autoencoder_transformer_GLOBAL.pth")
    print("[INFO] Modelo guardado en autoencoder_transformer_GLOBAL.pth")

    # ============================================================
    # VALIDACIÓN GLOBAL
    # ============================================================
    print("[INFO] Evaluando...")

    results = []

    # --- NO FRAUDE (validación) ---
    for f in val_nof_files:
        err = reconstruction_error(model, f, device)
        results.append([os.path.basename(f), err, 0])

    # --- FRAUDE ---
    for f in files_f:
        err = reconstruction_error(model, f, device)
        results.append([os.path.basename(f), err, 1])

    df = pd.DataFrame(results, columns=["archivo", "error", "clase"])
    df.to_csv("errores_transformer_GLOBAL_por_archivo.csv", index=False)
    print("[INFO] CSV guardado en errores_transformer_GLOBAL_por_archivo.csv")

    # ============================================================
    # PLOT
    # ============================================================
    plt.figure(figsize=(10,6))
    plt.hist(df[df.clase==0].error, bins=50, alpha=0.6, label="No Fraude")
    plt.hist(df[df.clase==1].error, bins=50, alpha=0.6, label="Fraude")
    plt.legend()
    plt.title("Distribución de errores reconstrucción")
    plt.xlabel("Error")
    plt.ylabel("Frecuencia")
    plt.savefig("hist_global.png")
    print("[INFO] Histograma guardado en hist_global.png")


if __name__ == "__main__":
    main()
