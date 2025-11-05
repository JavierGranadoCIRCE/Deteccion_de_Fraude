import pandas as pd
from collections import Counter

CSV_FILE = "output_5_limpio.csv"
CHUNKSIZE = 500_000
CUPS_COL = "CUPS"
DATE_COL = "FECHA"

total_rows = 0
cups_counter = Counter()

print("Analizando CSV grande (modo seguro)...")

for chunk in pd.read_csv(
        CSV_FILE,
        sep=';',
        chunksize=CHUNKSIZE,
        encoding='utf-8',
        dtype={CUPS_COL: str, DATE_COL: str},
        on_bad_lines='skip',
        engine='python'  # fuerza delimitador correcto aunque haya ruido
):
    total_rows += len(chunk)
    cups_counter.update(chunk[CUPS_COL].dropna().values)

print(f"\n✅ Total de filas: {total_rows:,}")
print(f"✅ Total de CUPS únicos: {len(cups_counter):,}")

counts = list(cups_counter.values())
avg_rows = sum(counts) / len(counts)
min_rows = min(counts)
max_rows = max(counts)

print(f"🔹 Filas promedio por contador: {avg_rows:.1f}")
print(f"🔹 Mínimo: {min_rows}, Máximo: {max_rows}")

least = sorted(cups_counter.items(), key=lambda x: x[1])[:10]
print("\n🔸 Ejemplo de CUPS con menos filas:")
for cups, n in least:
    print(f"  {cups}: {n}")


