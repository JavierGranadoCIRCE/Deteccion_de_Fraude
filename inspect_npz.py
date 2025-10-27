#!/usr/bin/env python3
"""
inspect_npz.py
--------------
Inspecciona visualmente un archivo .npz generado por el preprocesado de contadores.

Uso:
  python inspect_npz.py --file ruta/al/archivo.npz [--window N]

Ejemplo:
  python inspect_npz.py --file salida_npz/meter_000001.npz --window 5
"""

import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Ruta al archivo .npz")
    ap.add_argument("--window", type=int, default=0, help="Número de ventana a visualizar (por defecto 0)")
    args = ap.parse_args()

    data = np.load(args.file, allow_pickle=True)
    print("Claves disponibles:", data.files)

    X = data["X"]
    meta = data["meta"]
    try:
        meta = json.loads(str(meta))
    except Exception:
        meta = meta.item() if hasattr(meta, "item") else meta

    print("\n=== Información del archivo ===")
    print(f"Forma de X: {X.shape} (num_windows, L, 2)")
    print(f"Label (fraude): {meta.get('label', 'N/A')}")
    print(f"Ventanas: {meta.get('num_windows', 'N/A')}  |  L={meta.get('L', 'N/A')}  S={meta.get('S', 'N/A')}")
    print(f"Scaler: {meta.get('scaler', {})}")
    if "start_date" in meta:
        print(f"Fecha de inicio: {meta['start_date']}")

    # Seleccionar la ventana a mostrar
    idx = args.window
    if idx < 0 or idx >= X.shape[0]:
        print(f"⚠️ Ventana {idx} fuera de rango. Debe estar entre 0 y {X.shape[0]-1}.")
        return

    print(f"\nMostrando ventana {idx} de {X.shape[0]-1}...")
    plt.figure(figsize=(10, 4))
    plt.plot(X[idx, :, 0], label=f"Consumo normalizado (ventana {idx})")
    plt.plot(X[idx, :, 1] * X[idx,:,0].max(), 'r--', label="Tamper (escalado)")
    plt.legend()
    plt.title(f"Ventana {idx} - Label={meta.get('label', '?')}")
    plt.xlabel("Timestep (horas dentro de la ventana)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
