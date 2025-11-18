import os
import sys
import numpy as np
import pandas as pd

def cargar_lista_fraudes(csv_path):
    """Carga los nombres de CUPS fraudulentos desde el CSV."""
    try:
        df = pd.read_csv(csv_path, header=None)
        fraudes = set(df[0].astype(str).str.strip())
        print(f"✅ Cargados {len(fraudes)} CUPS fraudulentos desde '{csv_path}'.")
        return fraudes
    except Exception as e:
        print(f"❌ Error al leer {csv_path}: {e}")
        sys.exit(1)

def procesar_npz(carpeta, fraudes):
    """Abre cada .npz, añade la etiqueta de fraude y lo sobrescribe."""
    archivos = [f for f in os.listdir(carpeta) if f.endswith(".npz")]
    if not archivos:
        print("⚠️ No se encontraron archivos .npz en la carpeta especificada.")
        return

    total = len(archivos)
    modificados = 0
    for i, nombre in enumerate(archivos, 1):
        cups = os.path.splitext(nombre)[0]
        ruta = os.path.join(carpeta, nombre)
        try:
            data = np.load(ruta)
            energia = data["energia"]
            fallo = data["fallo"]
            combinado = data["combinado"]

            # Etiqueta de fraude
            etiqueta = np.array([1 if cups in fraudes else 0], dtype=np.int8)

            # Concatenar al final del array combinado
            combinado_nuevo = np.concatenate([combinado, etiqueta])

            # Guardar sobrescribiendo
            np.savez(ruta, energia=energia, fallo=fallo, combinado=combinado_nuevo)
            modificados += 1

        except Exception as e:
            print(f"⚠️ Error procesando {nombre}: {e}")

        if i % 100 == 0 or i == total:
            print(f"🧩 Procesados {i}/{total} archivos...", end="\r")

    print(f"\n\n✅ Proceso completado.")
    print(f"   → Archivos procesados: {total}")
    print(f"   → Archivos modificados: {modificados}")
    print(f"   → Cada .npz ahora contiene 17521 valores en el array 'combinado'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python añade_etiqueta_fraude.py <carpeta_con_npz>")
        sys.exit(1)

    carpeta = sys.argv[1]
    csv_fraudes = os.path.join(carpeta, "cups_fraudulentos.csv")

    if not os.path.exists(csv_fraudes):
        print(f"❌ No se encontró el archivo '{csv_fraudes}'. Debe contener la lista de CUPS fraudulentos.")
        sys.exit(1)

    fraudes = cargar_lista_fraudes(csv_fraudes)
    procesar_npz(carpeta, fraudes)
