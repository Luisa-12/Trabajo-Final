# Figura: distribucion de la similitud de los 190 pares para cada algoritmo,
# con el par del caso de estudio (11-16) marcado.
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corpus import cargar_corpus
from similitud.comparador import preparar_idf, comparar_par, NOMBRES

corpus = cargar_corpus()
idf = preparar_idf(corpus)
valores = {c: [] for c, _, _ in NOMBRES}
caso = {}
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        r = comparar_par(corpus[i], corpus[j], idf)
        for c in valores:
            valores[c].append(r[c]["similitud"])
            if corpus[i]["id"] == 11 and corpus[j]["id"] == 16:
                caso[c] = r[c]["similitud"]
etiquetas = ["Levenshtein", "Needleman-\nWunsch", "Coseno\nTF-IDF", "Jaccard", "Vectores de\npalabras", "Sentence-\nBERT"]
fig, ax = plt.subplots(figsize=(9, 4.8))
datos = [valores[c] for c, _, _ in NOMBRES]
bp = ax.boxplot(datos, tick_labels=etiquetas, patch_artist=True, widths=0.55)
for k, caja in enumerate(bp["boxes"]):
    caja.set_facecolor("#dbeafe" if k < 4 else "#dcfce7")
for k, (c, _, _) in enumerate(NOMBRES):
    ax.plot(k + 1, caso[c], marker="D", color="#dc2626", markersize=7, zorder=5,
            label="Par 11-16" if k == 0 else None)
ax.set_ylabel("Similitud")
ax.set_title("Distribucion de la similitud en los 190 pares del corpus")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "docs", "resultados", "distribucion_similitud.png"), dpi=150)
print("ok")
