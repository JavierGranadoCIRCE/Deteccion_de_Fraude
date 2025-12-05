import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv

# ============================================
#   MODELO COMPATIBLE CON EL ENTRENAMIENTO FAST
# ============================================
class AutoencoderSmall(nn.Module):
    def __init__(self, seq_len=168, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 2)
        )
        self.seq_len = seq_len

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out


# ============================================
#   DATASET PARA VENTANAS
# ============================================
class VentanasDataset(Dataset):
    def __init__(self, archivos, ventana):
        self.ventana = ventana
        self.X = []
        for f in archivos:
            d = np.load(f)
            energia = d["energia"]
            fallo = d["fallo"]

            serie = np.stack([energia, fallo], axis=1)  # (17520, 2)

            for i in range(0, len(serie) - ventana):
                self.X.append(serie[i:i+ventana])

        print(f"[INFO] Total ventanas generadas: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)


# ============================================
#   EVALUACIÓN
# ============================================
def evaluar(modelo, loader, device):
    model.eval()
    errores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluando"):
            batch = batch.to(device)
            pred = modelo(batch)
            loss = torch.mean((pred - batch) ** 2, dim=(1,2))
            errores.extend(loss.cpu().numpy().tolist())

    return errores


# ============================================
#   MAIN
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, required=True)
    parser.add_argument("--data_no_fraude", type=str, required=True)
    parser.add_argument("--data_fraude", type=str, required=True)
    parser.add_argument("--ventana", type=int, default=168)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # -------------------------
    # Cargar modelo
    # -------------------------
    print("[INFO] Cargando modelo...")
    model = AutoencoderSmall(seq_len=args.ventana)
    state = torch.load(args.modelo, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print("[OK] Modelo cargado correctamente")

    # -------------------------
    # Cargar archivos
    # -------------------------
    archivos_nf = [os.path.join(args.data_no_fraude, f)
                   for f in os.listdir(args.data_no_fraude) if f.endswith(".npz")]

    archivos_f = [os.path.join(args.data_fraude, f)
                  for f in os.listdir(args.data_fraude) if f.endswith(".npz")]

    print(f"[INFO] NO FRAUDE: {len(archivos_nf)} archivos")
    print(f"[INFO] FRAUDE: {len(archivos_f)} archivos")

    # -------------------------
    # Dataset / Dataloader
    # -------------------------
    ds_nf = VentanasDataset(archivos_nf, args.ventana)
    ds_f = VentanasDataset(archivos_f, args.ventana)

    loader_nf = DataLoader(ds_nf, batch_size=args.batch_size, shuffle=False)
    loader_f = DataLoader(ds_f, batch_size=args.batch_size, shuffle=False)

    # -------------------------
    # Evaluar
    # -------------------------
    print("\n[INFO] Evaluando NO FRAUDE...")
    errores_nf = evaluar(model, loader_nf, device)

    print("\n[INFO] Evaluando FRAUDE...")
    errores_f = evaluar(model, loader_f, device)

    # Guardar resultados
    np.savez("errores_autoencoder_small.npz",
             no_fraude=np.array(errores_nf),
             fraude=np.array(errores_f))

    print("\n[OK] Resultados guardados como errores_autoencoder_small.npz")
    print("[FIN]")
