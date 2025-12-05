import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# -------------------------------------------------------------
#   MODELO — EXACTAMENTE EL MISMO QUE train_autoencoder_unsupervised_fast.py
# -------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class AutoencoderTransformer(nn.Module):
    def __init__(self, seq_len=128, features=2, dim=128, layers=2):
        super().__init__()
        self.input_projection = nn.Linear(features, dim)
        self.encoder_layers = nn.ModuleList([TransformerBlock(dim) for _ in range(layers)])
        self.decoder_layers = nn.ModuleList([TransformerBlock(dim) for _ in range(layers)])
        self.output_projection = nn.Linear(dim, features)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.output_projection(x)
        return x

# -------------------------------------------------------------
#   FUNCIONES DE UTILIDAD
# -------------------------------------------------------------
def load_npz_file(path):
    data = np.load(path)
    energia = data["energia"]
    fallo = data["fallo"]
    combinado = np.vstack([energia, fallo]).T    # shape (17520, 2)
    return combinado.astype(np.float32)

def generar_ventanas(secuencia, ventana=128):
    total = len(secuencia)
    n = total // ventana
    secuencia = secuencia[:n * ventana]
    ventanas = secuencia.reshape(n, ventana, 2)
    return ventanas

def evaluar_directorio(model, directorio, ventana, batch_size, device):
    errores = []
    archivos = sorted(os.listdir(directorio))
    print(f"[INFO] Procesando {len(archivos)} archivos en: {directorio}")

    for f in tqdm(archivos):
        path = os.path.join(directorio, f)
        seq = load_npz_file(path)
        ventanas = generar_ventanas(seq, ventana)

        ds = torch.tensor(ventanas, device=device)
        dataset_size = len(ds)

        with torch.no_grad():
            for i in range(0, dataset_size, batch_size):
                batch = ds[i:i+batch_size]
                with torch.cuda.amp.autocast():
                    recon = model(batch)
                    loss = torch.mean((recon - batch)**2, dim=(1,2))
                errores.extend(loss.cpu().numpy())

    return np.array(errores)

# -------------------------------------------------------------
#   MAIN
# -------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Evaluando en", device)

    # Cargar modelo EXACTO
    model = AutoencoderTransformer(seq_len=args.ventana)
    state = torch.load(args.modelo, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("[INFO] Modelo cargado correctamente")

    # ---------------------------
    # Evaluar NO FRAUDE
    # ---------------------------
    print("\n[INFO] Procesando NO FRAUDE…")
    errores_no_fraude = evaluar_directorio(
        model, args.data_no_fraude, args.ventana, args.batch_size, device
    )

    # ---------------------------
    # Evaluar FRAUDE
    # ---------------------------
    print("\n[INFO] Procesando FRAUDE…")
    errores_fraude = evaluar_directorio(
        model, args.data_fraude, args.ventana, args.batch_size, device
    )

    # Guardar CSV
    np.savetxt("errores_autoencoder.csv",
               np.vstack([errores_no_fraude, errores_fraude]),
               delimiter=",")
    print("[OK] Guardado errores_autoencoder.csv")

    # Histograma
    plt.figure(figsize=(8,4))
    plt.hist(errores_no_fraude, bins=100, alpha=0.7, label="NO FRAUDE")
    plt.hist(errores_fraude, bins=100, alpha=0.7, label="FRAUDE")
    plt.legend()
    plt.title("Histograma de errores (Reconstrucción)")
    plt.savefig("hist_errores.png")
    print("[OK] Guardado hist_errores.png")

    # -----------------------------------------------------
    # MÉTRICAS
    # -----------------------------------------------------
    print("\n=== MÉTRICAS ===")
    y_true = np.concatenate([
        np.zeros_like(errores_no_fraude),
        np.ones_like(errores_fraude)
    ])
    y_score = np.concatenate([errores_no_fraude, errores_fraude])

    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    print(f"AUC-ROC = {auc_roc:.4f}")
    print(f"AUC-PR  = {auc_pr:.4f}")

    print("\n[FIN] Evaluación completada.")


# -------------------------------------------------------------
#   PARÁMETROS
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--data_no_fraude", required=True)
    parser.add_argument("--data_fraude", required=True)
    parser.add_argument("--ventana", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)
