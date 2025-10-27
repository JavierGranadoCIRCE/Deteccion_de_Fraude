#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
EXPECTED_T = 17520

def robust_scale(x):
    med = np.median(x[~np.isnan(x)])
    q25 = np.percentile(x[~np.isnan(x)], 25)
    q75 = np.percentile(x[~np.isnan(x)], 75)
    iqr = max(q75 - q25, 1e-6)
    return (x - med) / iqr, {"median": float(med), "q25": float(q25), "q75": float(q75), "iqr": float(iqr)}
def parse_row_values(values):
    if len(values) != EXPECTED_T * 2 + 1:
        raise ValueError(f"Fila tiene longitud {len(values)}, esperaba {EXPECTED_T*2+1} valores.")
    cons = np.array(values[0:EXPECTED_T], dtype=float)
    tamper = np.array(values[EXPECTED_T:EXPECTED_T*2], dtype=int)
    label = int(values[-1])
    return cons, tamper, label
def build_windows(cons, tamper, L, S):
    cons_norm, scaler = robust_scale(cons)
    T = len(cons_norm)
    X_list = []
    start_idx = []
    for start in range(0, T - L + 1, S):
        end = start + L
        X_list.append(np.stack([cons_norm[start:end], tamper[start:end]], axis=-1))
        start_idx.append(start)
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, L, 2), dtype=float)
    meta = {"L": L, "S": S, "num_windows": int(X.shape[0]), "scaler": scaler, "start_indices": start_idx}
    return X, meta
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--window", type=int, default=168)
    ap.add_argument("--stride", type=int, default=168)
    args = ap.parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    with open(csv_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if args.row >= len(lines):
        raise IndexError("Row index fuera de rango.")
    row = lines[args.row]
    parts = row.split(",")
    cons, tamper, label = parse_row_values(parts)
    X, meta = build_windows(cons, tamper, L=args.window, S=args.stride)
    meta["label"] = int(label)
    if args.start:
        meta["start_date"] = args.start
    np.savez_compressed(out_path, X=X, meta=json.dumps(meta))
    print(f"[OK] Guardado: {out_path}")
    print(f"   - Ventanas: {meta['num_windows']}  | L={meta['L']}  S={meta['S']}")
    print(f"   - Forma X: {X.shape}  (num_windows, L, 2)")
    print(f"   - Label contador: {meta['label']}")
    if meta['num_windows'] > 0:
        print(f"   - Primera ventana start_idx: {meta['start_indices'][0]}")
if __name__ == '__main__':
    main()
