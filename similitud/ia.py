# Similitud basada en embeddings (representaciones vectoriales densas).
#
# Modelo 1 - Vectores de palabras preentrenados (familia Word2Vec / GloVe / FastText):
#   cada palabra tiene un vector fijo de 300 dimensiones (spaCy en_core_web_md).
#   El vector del abstract es el PROMEDIO de los vectores de sus palabras.
#
# Modelo 2 - Sentence-BERT (all-MiniLM-L6-v2):
#   un transformer que lee el abstract completo y produce un vector de 384
#   dimensiones que tiene en cuenta el contexto de cada palabra.
#   Los vectores de los 20 abstracts se calcularon una sola vez con
#   scripts/generar_embeddings.py y se guardaron en datos/embeddings_minilm.json
#   (el corpus es estatico, asi que no hace falta cargar el modelo en la app).
#
# La similitud (coseno) y la distancia euclidiana se calculan aqui a mano.

import os
import json
import math
from preprocesamiento import tokenizar, quitar_stopwords

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vectores_palabras = None
_embeddings_minilm = None


def cargar_vectores_palabras():
    global _vectores_palabras
    if _vectores_palabras is None:
        with open(os.path.join(BASE, "datos", "vectores_palabras.json"), encoding="utf-8") as f:
            _vectores_palabras = json.load(f)
    return _vectores_palabras


def cargar_embeddings_minilm():
    global _embeddings_minilm
    if _embeddings_minilm is None:
        with open(os.path.join(BASE, "datos", "embeddings_minilm.json"), encoding="utf-8") as f:
            _embeddings_minilm = json.load(f)
    return _embeddings_minilm


# --------------------------- operaciones vectoriales ---------------------------
def producto_punto(u, v):
    s = 0.0
    for i in range(len(u)):
        s += u[i] * v[i]   # producto punto, componente por componente
    return s


def norma(u):
    return math.sqrt(producto_punto(u, u))


def coseno(u, v):
    nu = norma(u)
    nv = norma(v)
    if nu == 0 or nv == 0:
        return 0.0
    return producto_punto(u, v) / (nu * nv)    # coseno = punto / (largo A x largo B)


def distancia_euclidiana(u, v):
    s = 0.0
    for i in range(len(u)):
        s += (u[i] - v[i]) ** 2    # distancia euclidiana: suma de diferencias al cuadrado
    return math.sqrt(s)


def angulo_grados(cos):
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# --------------------------- modelo 1: promedio de vectores ---------------------------
def vector_promedio(texto):
    """Devuelve (vector promedio, palabras usadas, palabras sin vector)."""
    datos = cargar_vectores_palabras()
    vectores = datos["vectores"]
    dim = datos["dimension"]
    suma = [0.0] * dim
    usadas = []
    sin_vector = []
    for palabra in quitar_stopwords(tokenizar(texto)):
        if palabra in vectores:
            v = vectores[palabra]
            for i in range(dim):
                suma[i] += v[i]     # sumo el vector de cada palabra
            usadas.append(palabra)
        else:
            sin_vector.append(palabra)   # palabra que el modelo no conoce (ej. chatgpt)
    if len(usadas) == 0:
        return suma, usadas, sin_vector
    promedio = []
    for i in range(dim):
        promedio.append(suma[i] / len(usadas))  # vector del abstract = promedio de sus palabras
    return promedio, usadas, sin_vector


def similitud_vectores_palabras(texto_a, texto_b):
    va, _, _ = vector_promedio(texto_a)
    vb, _, _ = vector_promedio(texto_b)
    return coseno(va, vb), distancia_euclidiana(va, vb)


# --------------------------- modelo 2: Sentence-BERT ---------------------------
def embedding_minilm(id_articulo):
    return cargar_embeddings_minilm()["embeddings"][str(id_articulo)]


def similitud_minilm(id_a, id_b):
    ea = embedding_minilm(id_a)  # vector de Sentence-BERT ya calculado
    eb = embedding_minilm(id_b)
    return coseno(ea, eb), distancia_euclidiana(ea, eb)   # similitud y distancia entre los dos abstracts
