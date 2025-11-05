import numpy as np
import csv
from tqdm import trange

# ============================================================
#  GENERADOR DE DATOS FAKE PARA DETECCIÓN DE FRAUDE ELÉCTRICO
# ============================================================
# Cada fila del CSV contendrá:
#   17520 consumos horarios  (2 años)
#   17520 valores tamper     (0 o 1)
#   1 etiqueta final         (1 = fraude, 0 = no fraude)
# Total columnas = 35041
#
# El fichero resultante podrá usarse como entrada para:
#   batch_single_row_to_npz.py
# ============================================================

N_METERS = 6     # Número de contadores a generar
T = 17520             # Horas (2 años)
FRAUD_RATIO = 0.05    # % de contadores fraudulentos
OUT_FILE = "fake_meters_multirow_6.csv"
SEED = 42

rng = np.random.default_rng(SEED)

def generate_single(fraud=False):
    """Genera los datos de un contador: consumo, tamper, label"""
    t = np.arange(T)

    # --- Consumo base (2 kWh medio) con oscilaciones diarias y semanales ---
    baseline = 2.0 + 0.3 * np.sin(2 * np.pi * t / 24) + 0.2 * np.sin(2 * np.pi * t / (24 * 7))
    noise = rng.normal(0, 0.15, T)
    consumption = np.clip(baseline + noise, 0, None)

    # --- Tamper: 0 = tapa cerrada, 1 = tapa abierta ---
    tamper = np.zeros(T, dtype=int)

    if fraud:
        # entre 1 y 3 episodios de fraude
        for _ in range(rng.integers(1, 4)):
            start = rng.integers(0, T - 24 * 7)
            length = rng.integers(6, 24 * 7)  # entre 6 y 168 horas
            tamper[start:start + length] = 1
            # reducir consumo durante el fraude
            consumption[start:start + length] *= rng.uniform(0.2, 0.7)
    else:
        # contadores normales con pocos eventos de tamper sueltos
        for _ in range(rng.integers(0, 3)):
            s = rng.integers(0, T - 24)
            tamper[s:s + rng.integers(1, 6)] = 1

    label = int(fraud)
    return consumption, tamper, label


with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    for i in trange(N_METERS, desc="Generando contadores"):
        fraud = rng.random() < FRAUD_RATIO
        cons, tamp, label = generate_single(fraud)
        row = np.concatenate([cons, tamp, [label]])
        row = np.round(row, 4)  # 🔹 redondeo a 4 decimales
        writer.writerow(row)

print(f"\nCSV generado: {OUT_FILE} con {N_METERS} filas y {2*T+1} columnas")
