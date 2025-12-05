import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from comet_ml import Experiment
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
#  MODELO (igual que el de entrenamiento)
# ============================================================

class AutoencoderTransformer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, num_heads=2):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.final_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return self.final_layer(decoded)

# ============================================================
#  GENERACIÓN DE VENTANAS (igual que en entrenamiento)
# ============================================================

def generar_ventanas(energia, fallo, ventana):
    X = []
    n = len(energia)
    for i in range(0, n - ventana + 1, ventana):
        e = energia[i:i+ventana]
        f = fallo[i:i+ventana]
        matriz = np.stack([e, f], axis=1)  # (168, 2)
        X.append(matriz.astype(np.float32))
    return np.array(X)


# ============================================================
#  CARGA DE DATOS NPZ Y CÁLCULO DE ERRORES
# ============================================================

def evaluar_directorio(directorio, ventana, model, device):
    resultados = []

    archivos = sorted([f for f in os.listdir(directorio) if f.endswith(".npz")])

    for fname in tqdm(archivos, desc=f"Evaluando {directorio}"):
        path = os.path.join(directorio, fname)
        data = np.load(path)

        energia = data["energia"].astype(np.float32)
        fallo = data["fallo"].astype(np.float32)

        ventanas = generar_ventanas(energia, fallo, ventana)

        if ventanas.shape[0] == 0:
            continue

        batch = torch.tensor(ventanas).to(device)
        with torch.no_grad():
            out = model(batch)
            mse = torch.mean((out - batch)**2, dim=(1,2)).cpu().numpy()

        resultados.append({
            "archivo": fname,
            "error_mean": float(np.mean(mse)),
            "error_p95": float(np.percentile(mse, 95)),
            "n_ventanas": ventanas.shape[0]
        })

    return resultados


# ============================================================
#  BÚSQUEDA DE UMBRAL ÓPTIMO (minimizar FP)
# ============================================================

def buscar_mejor_umbral(df):
    percentiles = np.linspace(90, 99.9, 100)
    candidatos = []

    for p in percentiles:
        umbral = np.percentile(df[df["clase"] == 0]["error_mean"], p)

        TP = ((df["error_mean"] >= umbral) & (df["clase"] == 1)).sum()
        FP = ((df["error_mean"] >= umbral) & (df["clase"] == 0)).sum()
        TN = ((df["error_mean"] < umbral) & (df["clase"] == 0)).sum()
        FN = ((df["error_mean"] < umbral) & (df["clase"] == 1)).sum()

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        fpr = FP / (FP + TN + 1e-9)

        candidatos.append([p, umbral, TP, FP, TN, FN, precision, recall, f1, fpr])

    df_cand = pd.DataFrame(candidatos,
                           columns=["percentil", "umbral", "TP", "FP", "TN", "FN", "precision", "recall", "f1", "fpr"])

    df_sorted = df_cand.sort_values("FP")
    mejor = df_sorted.iloc[0]

    return mejor, df_cand


# ============================================================
#  MAIN
# ============================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Evaluando en {device}")

    experiment = Experiment(
        api_key="st7c3hq16vGAMOgB2jApUCgN0",
        project_name="fraude-autoencoder-ventanas",
        workspace="javier-granado"
    )

    # Cargar modelo
    model = AutoencoderTransformer().to(device)
    model.load_state_dict(torch.load(args.modelo, map_location=device))
    model.eval()

    print("\n[INFO] Evaluando NO FRAUDE…")
    res_no_fraude = evaluar_directorio(args.data_no_fraude, args.ventana, model, device)
    print("[OK] NO FRAUDE evaluado.")

    print("\n[INFO] Evaluando FRAUDE…")
    res_fraude = evaluar_directorio(args.data_fraude, args.ventana, model, device)
    print("[OK] FRAUDE evaluado.")

    # Convertir a DataFrame
    df_no = pd.DataFrame(res_no_fraude)
    df_no["clase"] = 0

    df_fra = pd.DataFrame(res_fraude)
    df_fra["clase"] = 1

    df = pd.concat([df_no, df_fra])
    df.to_csv("resultados_autoencoder_ventanas.csv", index=False)

    # ===========================
    # Buscar mejor umbral
    # ===========================
    mejor, df_cand = buscar_mejor_umbral(df)

    print("\n=============== UMBRAL RECOMENDADO ===============")
    print(mejor)
    print("==================================================")

    df_cand.to_csv("metricas_umbral_autoencoder_ventanas.csv", index=False)

    # ===========================
    # Histograma
    # ===========================
    plt.figure(figsize=(10,5))
    plt.hist(df[df["clase"]==0]["error_mean"], bins=100, alpha=0.7, label="No fraude")
    plt.hist(df[df["clase"]==1]["error_mean"], bins=100, alpha=0.7, label="Fraude")
    plt.legend()
    plt.title("Errores reconstrucción (mean por cliente)")
    plt.xlabel("MSE")
    plt.ylabel("Frecuencia")
    plt.savefig("histograma_ventanas.png")

    experiment.log_image("histograma_ventanas.png")

    print("\n[OK] Evaluación completada.")
    print("[OK] Archivos generados:")
    print("    - resultados_autoencoder_ventanas.csv")
    print("    - metricas_umbral_autoencoder_ventanas.csv")
    print("    - histograma_ventanas.png\n")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--data_no_fraude", required=True)
    parser.add_argument("--data_fraude", required=True)
    parser.add_argument("--ventana", type=int, default=168)
    args = parser.parse_args()
    main(args)
