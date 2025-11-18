import os
import sys
import csv

def comparar_cups_y_npz(base_dir, archivo_csv):
    # --- Leer CUPS del CSV ---
    with open(archivo_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        cups_csv = set(row[0].strip().upper().replace(".CSV", "").replace(".NPZ", "") for row in reader if row)

    print(f"📄 CUPS en CSV: {len(cups_csv)}")

    # --- Recorrer todos los .npz ---
    cups_npz = set()
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".npz"):
                cups_npz.add(os.path.splitext(f)[0].upper())

    print(f"📦 CUPS en archivos NPZ: {len(cups_npz)}")

    # --- Comparar ---
    interseccion = cups_csv & cups_npz
    faltan = cups_csv - cups_npz
    extras = cups_npz - cups_csv

    print(f"\n✅ Coinciden en ambos: {len(interseccion)}")
    print(f"❌ Faltan en NPZ:      {len(faltan)}")
    print(f"⚠️  Sobran en NPZ:      {len(extras)}")

    if faltan:
        print("\nEjemplo de CUPS del CSV que no tienen NPZ:")
        for c in list(faltan)[:20]:
            print("  ", c)
    if extras:
        print("\nEjemplo de NPZ sin correspondencia en el CSV:")
        for c in list(extras)[:20]:
            print("  ", c)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python compara_cups_npz.py <carpeta_npz> <cups_fraudulentos.csv>")
        sys.exit(1)

    carpeta_base = sys.argv[1]
    archivo_csv = sys.argv[2]
    comparar_cups_y_npz(carpeta_base, archivo_csv)
