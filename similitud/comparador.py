# Ejecuta los 6 algoritmos de similitud sobre los articulos seleccionados.

import time
from preprocesamiento import preprocesar
from similitud import clasicos, ia

NOMBRES = [
    ("levenshtein", "Levenshtein (palabras)", "clasico"),
    ("needleman", "Needleman-Wunsch (palabras)", "clasico"),
    ("coseno_tfidf", "Coseno TF-IDF", "clasico"),
    ("jaccard", "Jaccard", "clasico"),
    ("vectores_palabras", "Promedio de vectores de palabras (300d)", "ia"),
    ("minilm", "Sentence-BERT all-MiniLM-L6-v2 (384d)", "ia"),
]


def preparar_idf(corpus):
    documentos = []
    for art in corpus:
        documentos.append(preprocesar(art["abstract"]))
    idf, df = clasicos.calcular_idf(documentos)
    return idf


def comparar_par(art_a, art_b, idf):
    """Devuelve un diccionario {algoritmo: {"similitud":..., "tiempo_ms":..., ...}}"""
    ta = preprocesar(art_a["abstract"])
    tb = preprocesar(art_b["abstract"])
    resultado = {}

    inicio = time.perf_counter()
    s = clasicos.similitud_levenshtein(ta, tb)
    resultado["levenshtein"] = {"similitud": s, "tiempo_ms": (time.perf_counter() - inicio) * 1000}

    inicio = time.perf_counter()
    s = clasicos.similitud_needleman_wunsch(ta, tb)
    resultado["needleman"] = {"similitud": s, "tiempo_ms": (time.perf_counter() - inicio) * 1000}

    inicio = time.perf_counter()
    s = clasicos.similitud_coseno_tfidf(ta, tb, idf)
    resultado["coseno_tfidf"] = {"similitud": s, "tiempo_ms": (time.perf_counter() - inicio) * 1000}

    inicio = time.perf_counter()
    s = clasicos.jaccard(ta, tb)
    resultado["jaccard"] = {"similitud": s, "tiempo_ms": (time.perf_counter() - inicio) * 1000}

    inicio = time.perf_counter()
    s, d = ia.similitud_vectores_palabras(art_a["abstract"], art_b["abstract"])
    resultado["vectores_palabras"] = {"similitud": s, "distancia": d, "tiempo_ms": (time.perf_counter() - inicio) * 1000}

    inicio = time.perf_counter()
    s, d = ia.similitud_minilm(art_a["id"], art_b["id"])
    resultado["minilm"] = {"similitud": s, "distancia": d, "tiempo_ms": (time.perf_counter() - inicio) * 1000}
    return resultado


def comparar_varios(articulos, idf):
    """Para 2 o mas articulos compara todos los pares posibles."""
    pares = []
    for i in range(len(articulos)):
        for j in range(i + 1, len(articulos)):
            pares.append((articulos[i], articulos[j], comparar_par(articulos[i], articulos[j], idf)))
    return pares
