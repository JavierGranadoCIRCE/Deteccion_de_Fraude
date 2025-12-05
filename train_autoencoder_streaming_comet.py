# ================================================================
# train_autoencoder_streaming_comet.py
# Entrenamiento rápido + eficiente del autoencoder usando ventanas
# con logging de Comet (solo LOSS)
# ================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from comet_ml import Experiment
from tqdm import tqdm
import time


# ================================================================
# MODELO AUTOENCODER (igual que en streaming anterior)
# ================================================================
class AutoencoderTransformer(nn.Module):
    def __init__(self, dim=64, heads=4, layers=2):
        super().__init__()

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=dim*4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(self.enc_layer, num_layers=layers)

        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=dim*4,
            batch_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(self.dec_layer, num_layers=layers)

        self.in_proj = nn.Linear(2, dim)
        self.out_proj = nn.Linear(dim, 2)

    def forward(self, x):
        h = self.in_proj(x)
        z = self.encoder(h)
        out = self.decoder(z, z)
        return self.out_proj(out)


# ================================================================
# DATASET STREAMING
# ================================================================
class VentanasDataset(Dataset):
    def __init__(self, folder, ventana=168, stride=24):
        self.files = sorted(os.listdir(folder))
        self.folder = folder
        self.ventana = ventana
        self.stride = stride
        self.index_map = []  # (file_index, start_pos)

        print("\n[INFO] Inicializando VentanasDataset (puede tardar un poco)...")

        # Construimos índices de forma streaming (sin cargar datos)
        for i_file, fname in enumerate(tqdm(self.files)):
            path = os.path.join(folder, fname)
            data = np.load(path)

            energia = data["energia"]
            fallo = data["fallo"]

            if len(energia) != 17520 or len(fallo) != 17520:
                continue

            length = len(energia)
            for start in range(0, length - ventana + 1, stride):
                self.index_map.append((i_file, start))

        print(f"[OK] Dataset construido con {len(self.index_map)} ventanas totales.\n")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, start = self.index_map[idx]
        fname = self.files[file_idx]
        full_path = os.path.join(self.folder, fname)

        data = np.load(full_path)
        energia = data["energia"]
        fallo = data["fallo"]

        ventana = np.stack([
            energia[start:start+self.ventana],
            fallo[start:start+self.ventana]
        ], axis=1).astype(np.float32)  # (ventana, 2)

        return torch.tensor(ventana)


# ================================================================
# ENTRENAMIENTO
# ================================================================
def train(
        data_dir,
        epochs=20,
        batch_size=512,
        ventana=168,
        stride=24
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Usando dispositivo: {device}")

    # -------------------------
    # COMET EXPERIMENTO
    # -------------------------
    experiment = Experiment(
        api_key="st7c3hq16vGAMOgB2jApUCgN0",
        project_name="fraude-autoencoder-streaming",
        workspace="javier-granado"
    )

    experiment.log_parameters({
        "ventana": ventana,
        "stride": stride,
        "batch_size": batch_size,
        "epochs": epochs,
        "data_dir": data_dir
    })

    # -------------------------
    # DATASET + DATALOADER
    # -------------------------
    dataset = VentanasDataset(data_dir, ventana=ventana, stride=stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # -------------------------
    # MODELO
    # -------------------------
    model = AutoencoderTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # -------------------------
    # LOOP DE ENTRENAMIENTO
    # -------------------------
    for epoch in range(1, epochs+1):

        start_time = time.time()
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}  |  Loss={avg:.6f}  |  Tiempo={elapsed:.1f}s")

        experiment.log_metric("loss", avg, epoch=epoch)
        experiment.log_metric("epoch_time_sec", elapsed, epoch=epoch)

        torch.save(model.state_dict(), f"autoencoder_streaming_epoch{epoch}.pth")

    torch.save(model.state_dict(), "autoencoder_streaming_FINAL.pth")
    print("\n✔ Entrenamiento completado. Modelo guardado como autoencoder_streaming_FINAL.pth")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--ventana", type=int, default=168)
    parser.add_argument("--stride", type=int, default=24)

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ventana=args.ventana,
        stride=args.stride
    )

