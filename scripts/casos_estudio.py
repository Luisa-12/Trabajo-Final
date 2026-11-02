# Genera los datos de los casos de estudio que se presentan en el documento tecnico.
#   python scripts/casos_estudio.py
# Salida: docs/resultados/casos_estudio.json y dendrogramas en PNG.

import os
import sys
import json
import math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from corpus import cargar_corpus, buscar_articulo
from preprocesamiento import preprocesar, tokenizar, quitar_stopwords
from similitud import clasicos, ia
from similitud.comparador import preparar_idf, comparar_par, NOMBRES
from clustering import jerarquico, dendrograma
from busqueda import kmp, boyer_moore, rabin_karp
from busqueda.buscador import a_minusculas

SALIDA = os.path.join(BASE, "docs", "resultados")
os.makedirs(SALIDA, exist_ok=True)
corpus = cargar_corpus()
idf = preparar_idf(corpus)
out = {}

# ============================ R1: articulos 11 y 16 ============================
A = buscar_articulo(corpus, 11)
B = buscar_articulo(corpus, 16)
ta = preprocesar(A["abstract"])
tb = preprocesar(B["abstract"])
r1 = {"id_a": 11, "id_b": 16, "n_terminos_a": len(ta), "n_terminos_b": len(tb),
      "terminos_a": ta, "terminos_b": tb}

# Levenshtein y NW sobre fragmento (6 palabras) y sobre el abstract completo
k = 6
d_frag, M = clasicos.levenshtein(ta[:k], tb[:k])
r1["lev_fragmento"] = {"a": ta[:k], "b": tb[:k], "matriz": M, "distancia": d_frag}
r1["lev_completo"] = {"distancia": clasicos.levenshtein_distancia(ta, tb), "similitud": clasicos.similitud_levenshtein(ta, tb)}
r1["lev_caracteres"] = {"n": len(A["abstract"]), "m": len(B["abstract"]),
                        "distancia": clasicos.levenshtein_distancia(A["abstract"].lower(), B["abstract"].lower()),
                        "similitud": clasicos.similitud_levenshtein(A["abstract"].lower(), B["abstract"].lower())}
p, F, al_a, al_b = clasicos.needleman_wunsch(ta[:k], tb[:k])
r1["nw_fragmento"] = {"matriz": F, "puntaje": p, "al_a": al_a, "al_b": al_b}
p, F, al_a, al_b = clasicos.needleman_wunsch(ta, tb)
coinc = sum(1 for i in range(len(al_a)) if al_a[i] == al_b[i])
r1["nw_completo"] = {"puntaje": p, "longitud_alineamiento": len(al_a), "coincidencias": coinc,
                     "huecos": sum(1 for i in range(len(al_a)) if al_a[i] == "-" or al_b[i] == "-"),
                     "similitud": coinc / len(al_a),
                     "palabras_alineadas": [al_a[i] for i in range(len(al_a)) if al_a[i] == al_b[i]]}

# TF-IDF
_, df = clasicos.calcular_idf([preprocesar(x["abstract"]) for x in corpus])
va = clasicos.vector_tfidf(ta, idf)
vb = clasicos.vector_tfidf(tb, idf)
tfa = clasicos.frecuencias(ta)
tfb = clasicos.frecuencias(tb)
comunes = []
for t in va:
    if t in vb:
        comunes.append({"termino": t, "tf_a": tfa[t], "tf_b": tfb[t], "df": df[t], "idf": idf[t],
                        "w_a": va[t], "w_b": vb[t], "producto": va[t] * vb[t]})
comunes.sort(key=lambda x: -x["producto"])
norma_a = math.sqrt(sum(v * v for v in va.values()))
norma_b = math.sqrt(sum(v * v for v in vb.values()))
punto = sum(c["producto"] for c in comunes)
r1["tfidf"] = {"N": len(corpus), "terminos_distintos_a": len(va), "terminos_distintos_b": len(vb),
               "comunes": comunes, "producto_punto": punto, "norma_a": norma_a, "norma_b": norma_b,
               "coseno": punto / (norma_a * norma_b), "angulo": ia.angulo_grados(punto / (norma_a * norma_b)),
               "top_a": sorted(va.items(), key=lambda x: -x[1])[:8], "top_b": sorted(vb.items(), key=lambda x: -x[1])[:8]}
# Jaccard
SA = set(ta)
SB = set(tb)
inter = sorted(SA & SB)
r1["jaccard"] = {"tam_a": len(SA), "tam_b": len(SB), "interseccion": inter, "tam_inter": len(inter),
                 "tam_union": len(SA | SB), "valor": len(inter) / len(SA | SB)}
# Vectores de palabras
vpa, usadas_a, sin_a = ia.vector_promedio(A["abstract"])
vpb, usadas_b, sin_b = ia.vector_promedio(B["abstract"])
cos_v = ia.coseno(vpa, vpb)
r1["vectores_palabras"] = {"usadas_a": len(usadas_a), "usadas_b": len(usadas_b), "sin_vector_a": sin_a, "sin_vector_b": sin_b,
                           "primeras_dim_a": vpa[:5], "primeras_dim_b": vpb[:5], "punto": ia.producto_punto(vpa, vpb),
                           "norma_a": ia.norma(vpa), "norma_b": ia.norma(vpb), "coseno": cos_v,
                           "angulo": ia.angulo_grados(cos_v), "euclidiana": ia.distancia_euclidiana(vpa, vpb)}
# MiniLM
ea = ia.embedding_minilm(11)
eb = ia.embedding_minilm(16)
cos_m = ia.coseno(ea, eb)
datos_m = ia.cargar_embeddings_minilm()
r1["minilm"] = {"primeras_dim_a": ea[:5], "primeras_dim_b": eb[:5], "norma_a": ia.norma(ea), "norma_b": ia.norma(eb),
                "coseno": cos_m, "angulo": ia.angulo_grados(cos_m), "euclidiana": ia.distancia_euclidiana(ea, eb),
                "tokens_a": datos_m.get("tokens_por_abstract", {}).get("11"), "tokens_b": datos_m.get("tokens_por_abstract", {}).get("16")}
r1["resumen"] = {k2: v["similitud"] for k2, v in comparar_par(A, B, idf).items()}

# Comparacion global sobre los 190 pares (para el analisis critico)
valores = {clave: [] for clave, _, _ in NOMBRES}
pares = []
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        res = comparar_par(corpus[i], corpus[j], idf)
        pares.append((corpus[i]["id"], corpus[j]["id"]))
        for clave in valores:
            valores[clave].append(res[clave]["similitud"])


def rangos(xs):
    orden = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    for pos, i in enumerate(orden):
        r[i] = pos + 1
    return r


def spearman(x, y):
    rx = rangos(x)
    ry = rangos(y)
    n = len(x)
    return 1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


estad = {}
for clave in valores:
    xs = valores[clave]
    media = sum(xs) / len(xs)
    desv = math.sqrt(sum((x - media) ** 2 for x in xs) / len(xs))
    mejor = max(range(len(xs)), key=lambda i: xs[i])
    estad[clave] = {"media": media, "desv": desv, "min": min(xs), "max": max(xs), "par_max": pares[mejor],
                    "posicion_11_16": sorted(xs, reverse=True).index(r1["resumen"][clave]) + 1}
corr = {}
for c1 in valores:
    corr[c1] = {c2: spearman(valores[c1], valores[c2]) for c2 in valores}
r1["global"] = {"estadisticas": estad, "spearman": corr, "n_pares": len(pares)}
out["r1"] = r1

# ============================ R2 ============================
r2 = {}
etiquetas = ["%d. %s" % (a["id"], a["titulo"][:48]) for a in corpus]
for rep, D in [("tfidf", jerarquico.matriz_distancias_tfidf(corpus)), ("minilm", jerarquico.matriz_distancias_minilm(corpus))]:
    r2[rep] = {}
    for metodo, titulo in [("single", "Single linkage"), ("complete", "Complete linkage"), ("average", "Average linkage")]:
        un = jerarquico.aglomerativo(D, metodo)
        ev = jerarquico.evaluar(D, un)
        grupos = jerarquico.cortar_arbol(un, len(corpus), 3)
        miembros = {}
        for i in range(len(corpus)):
            miembros.setdefault(grupos[i], []).append(corpus[i]["id"])
        nombre_rep = "TF-IDF" if rep == "tfidf" else "Sentence-BERT"
        dendrograma.dibujar(un, etiquetas, "%s - %s" % (titulo, nombre_rep), 3,
                            ruta=os.path.join(SALIDA, "dendrograma_%s_%s.png" % (rep, metodo)))
        r2[rep][metodo] = {"ccc": ev["ccc"], "silueta": ev["silueta"], "tamanos": ev["tamanos"],
                           "grupos_k3": list(miembros.values()),
                           "uniones": [(corpus[a]["id"] if a < 20 else "G%d" % a, corpus[b]["id"] if b < 20 else "G%d" % b, d, t)
                                       for a, b, d, t in un]}
    # distancias de ejemplo para la demostracion
    r2[rep]["D_ejemplo"] = [[round(D[i][j], 4) for j in [10, 15, 14, 17, 18]] for i in [10, 15, 14, 17, 18]]
out["r2"] = r2

# ============================ R3 ============================
texto = a_minusculas("Our results suggested that assessment should be transformed to cultivate students' self-regulated "
                     "learning skills, responsible learning, and integrity. To successfully transform assessment in higher education")
patron = "assessment"
r3 = {"texto": texto, "patron": patron, "n": len(texto), "m": len(patron)}
r3["kmp"] = {"pi": kmp.tabla_prefijos(patron)}
pos, comp, pasos = kmp.buscar(texto, patron, traza=True)
r3["kmp"].update({"posiciones": pos, "comparaciones": comp, "pasos": pasos})
r3["bm"] = {"malo": boyer_moore.tabla_caracter_malo(patron), "sufijo": boyer_moore.tabla_sufijo_bueno(patron)}
pos, comp, pasos = boyer_moore.buscar(texto, patron, traza=True)
r3["bm"].update({"posiciones": pos, "comparaciones": comp, "pasos": pasos})
# Rabin-Karp: Horner paso a paso con q = 101 y deteccion de colisiones con q = 13 y q = 101
q = 101
horner = []
h = 0
for c in patron:
    nuevo = (h * 256 + ord(c)) % q
    horner.append({"c": c, "ord": ord(c), "anterior": h, "nuevo": nuevo})
    h = nuevo
r3["rk"] = {"q": q, "horner": horner, "hash_patron": h, "h_mult": pow(256, len(patron) - 1, q)}
for qq in [13, 101, 1000000007]:
    pos, comp, pasos, col = rabin_karp.buscar(texto, patron, q=qq, traza=True)
    r3["rk"]["q_%d" % qq] = {"posiciones": pos, "comparaciones": comp, "colisiones": col,
                             "pasos": pasos if qq != 1000000007 else []}
# primeras ventanas con el rolling hash explicito (q = 101)
ventanas = []
ht = rabin_karp.hash_cadena(texto[:len(patron)], q=q)
hm = pow(256, len(patron) - 1, q)
for s in range(6):
    sig = (256 * (ht - ord(texto[s]) * hm) + ord(texto[s + len(patron)])) % q
    ventanas.append({"s": s, "ventana": texto[s:s + len(patron)], "hash": ht, "sale": texto[s], "entra": texto[s + len(patron)], "siguiente": sig})
    ht = sig
r3["rk"]["ventanas"] = ventanas
out["r3"] = r3

with open(os.path.join(SALIDA, "casos_estudio.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("Listo:", os.path.join(SALIDA, "casos_estudio.json"))
