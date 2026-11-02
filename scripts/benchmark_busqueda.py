# Experimentos de rendimiento para el analisis del Requerimiento 3.
# Genera docs/resultados/benchmark_busqueda.json y graficas PNG.
#   python scripts/benchmark_busqueda.py

import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corpus import cargar_corpus
from busqueda.buscador import a_minusculas
from busqueda.rendimiento import comparar, regresion_lineal

REP = 30  # repeticiones por medicion

SALIDA = os.path.join(BASE, "docs", "resultados")
os.makedirs(SALIDA, exist_ok=True)

corpus = cargar_corpus()
texto = ""
for art in corpus:
    texto = texto + a_minusculas(art["abstract"]) + "\n"
print("Longitud del texto (20 abstracts):", len(texto), "caracteres")

resultados = {"n_corpus": len(texto)}

# 1) Patrones de distinta longitud sobre el corpus real
patrones = ["ai", "data", "learning", "assessment", "systematic review",
            "artificial intelligence", "generative artificial intelligence",
            "artificial intelligence in education (aied)", "quantum computing"]
resultados["por_patron"] = []
for p in patrones:
    filas = comparar(texto, p, repeticiones=REP)
    resultados["por_patron"].append({"patron": p, "m": len(p), "filas": filas})
    print(p, [(f["algoritmo"][:5], round(f["total_ms"], 2), f["comparaciones"]) for f in filas])

# 2) Escalabilidad: el mismo texto repetido k veces (n crece, m fijo)
resultados["escalabilidad"] = []
patron = "generative artificial intelligence"
for k in [1, 2, 4, 8, 16]:
    t = texto * k
    filas = comparar(t, patron, repeticiones=REP)
    resultados["escalabilidad"].append({"n": len(t), "filas": filas})
    print("n =", len(t), [(f["algoritmo"][:5], round(f["total_ms"], 1)) for f in filas])

# 3) Peor caso teorico: texto "aaaa...a" y patron "aa...a" (todas las posiciones coinciden)
resultados["peor_caso"] = []
for m in [5, 10, 20, 40]:
    t = "a" * 20000
    filas = comparar(t, "a" * m, repeticiones=REP)
    resultados["peor_caso"].append({"m": m, "n": len(t), "filas": filas})
    print("peor caso m =", m, [(f["algoritmo"][:5], f["comparaciones"]) for f in filas])

# regresion lineal tiempo = a + b*n para comprobar el crecimiento lineal
resultados["regresion"] = {}
for alg in ["KMP", "Boyer-Moore", "Rabin-Karp", "Fuerza bruta (referencia)"]:
    xs = [r["n"] for r in resultados["escalabilidad"]]
    ys = [next(f for f in r["filas"] if f["algoritmo"] == alg)["total_ms"] for r in resultados["escalabilidad"]]
    a, b, r2 = regresion_lineal(xs, ys)
    resultados["regresion"][alg] = {"a": a, "b_ms_por_caracter": b, "r2": r2}
    print("regresion", alg, "b = %.3e ms/caracter, R2 = %.5f" % (b, r2))
resultados["repeticiones"] = REP

with open(os.path.join(SALIDA, "benchmark_busqueda.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------ graficas ------------------------------
COLOR = {"KMP": "#2563eb", "Boyer-Moore": "#dc2626", "Rabin-Karp": "#16a34a", "Fuerza bruta (referencia)": "#6b7280"}

fig, ax = plt.subplots(figsize=(8, 4.5))
for alg in COLOR:
    xs = [r["m"] for r in resultados["por_patron"]]
    ys = [next(f for f in r["filas"] if f["algoritmo"] == alg)["comparaciones"] for r in resultados["por_patron"]]
    ax.plot(xs, ys, marker="o", label=alg, color=COLOR[alg])
ax.set_xlabel("Longitud del patron m")
ax.set_ylabel("Comparaciones de caracteres")
ax.set_title("Comparaciones vs longitud del patron (n = %d)" % len(texto))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "comparaciones_vs_m.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for alg in COLOR:
    xs = [r["n"] for r in resultados["escalabilidad"]]
    ys = [next(f for f in r["filas"] if f["algoritmo"] == alg)["total_ms"] for r in resultados["escalabilidad"]]
    es = [next(f for f in r["filas"] if f["algoritmo"] == alg)["total_ic95"] for r in resultados["escalabilidad"]]
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=alg, color=COLOR[alg])
ax.set_xlabel("Longitud del texto n (caracteres)")
ax.set_ylabel("Tiempo medio (ms), barras = IC 95 %")
ax.set_title("Tiempo vs tamano del texto (patron de %d caracteres)" % len(patron))
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "tiempo_vs_n.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for alg in COLOR:
    xs = [r["m"] for r in resultados["peor_caso"]]
    ys = [next(f for f in r["filas"] if f["algoritmo"] == alg)["comparaciones"] for r in resultados["peor_caso"]]
    ax.plot(xs, ys, marker="o", label=alg, color=COLOR[alg])
ax.set_xlabel("Longitud del patron m")
ax.set_ylabel("Comparaciones de caracteres")
ax.set_title("Peor caso: texto 'a'*20000, patron 'a'*m")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "peor_caso.png"), dpi=150)
plt.close(fig)
print("Listo")
