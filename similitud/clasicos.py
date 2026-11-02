# Algoritmos clasicos de similitud textual implementados desde cero.
#  1. Distancia de Levenshtein (programacion dinamica)
#  2. Needleman-Wunsch (alineamiento global, programacion dinamica)
#  3. Similitud del coseno con pesos TF-IDF (modelo de espacio vectorial)
#  4. Coeficiente de Jaccard (conjuntos)
# Las secuencias pueden ser cadenas (caracter a caracter) o listas de palabras.

import math


# ---------------------------------------------------------------------------
# 1. Levenshtein
# ---------------------------------------------------------------------------
def levenshtein(a, b):
    """Devuelve (distancia, matriz). La matriz D tiene (n+1) x (m+1) celdas y
    D[i][j] es la distancia entre los primeros i elementos de a y los primeros j de b."""
    n = len(a)
    m = len(b)
    D = []
    for i in range(n + 1):
        fila = []
        for j in range(m + 1):
            fila.append(0)
        D.append(fila)
    # caso base: transformar en/desde la secuencia vacia
    for i in range(n + 1):
        D[i][0] = i     # primera columna: borrar i palabras
    for j in range(m + 1):
        D[0][j] = j     # primera fila: insertar j palabras
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                costo = 0
            else:
                costo = 1
            eliminar = D[i - 1][j] + 1   # vecina de arriba: borrar
            insertar = D[i][j - 1] + 1   # vecina de la izquierda: insertar
            sustituir = D[i - 1][j - 1] + costo   # diagonal: igual (0) o sustituir (1)
            D[i][j] = min(eliminar, insertar, sustituir)   # me quedo con la opción más barata
    return D[n][m], D    # la respuesta está en la última celda


def levenshtein_distancia(a, b):
    """Version que solo guarda dos filas: O(n*m) en tiempo y O(m) en memoria.
    Se usa cuando las secuencias son largas (abstracts completos por caracteres)."""
    n = len(a)
    m = len(b)
    anterior = list(range(m + 1))
    for i in range(1, n + 1):
        actual = [i] + [0] * m
        for j in range(1, m + 1):
            costo = 0 if a[i - 1] == b[j - 1] else 1
            actual[j] = min(anterior[j] + 1, actual[j - 1] + 1, anterior[j - 1] + costo)
        anterior = actual
    return anterior[m]


def similitud_levenshtein(a, b):
    """Normaliza la distancia en [0, 1]: 1 = identicos."""
    if len(a) == 0 and len(b) == 0:
        return 1.0
    d = levenshtein_distancia(a, b)
    return 1 - d / max(len(a), len(b))


# ---------------------------------------------------------------------------
# 2. Needleman-Wunsch
# ---------------------------------------------------------------------------
def needleman_wunsch(a, b, coincidencia=1, diferencia=-1, hueco=-1):
    """Alineamiento global. Devuelve (puntaje, matriz F, alineamiento_a, alineamiento_b).
    F[i][j] = mejor puntaje de alinear a[0..i-1] con b[0..j-1]."""
    n = len(a)
    m = len(b)
    F = []
    for i in range(n + 1):
        F.append([0] * (m + 1))
    for i in range(n + 1):
        F[i][0] = i * hueco
    for j in range(m + 1):
        F[0][j] = j * hueco
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                s = coincidencia
            else:
                s = diferencia
            diagonal = F[i - 1][j - 1] + s
            arriba = F[i - 1][j] + hueco      # hueco en b
            izquierda = F[i][j - 1] + hueco   # hueco en a
            F[i][j] = max(diagonal, arriba, izquierda)

    # reconstruccion del alineamiento (traceback) desde la esquina inferior
    al_a = []
    al_b = []
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = coincidencia if a[i - 1] == b[j - 1] else diferencia
            if F[i][j] == F[i - 1][j - 1] + s:
                al_a.append(a[i - 1])
                al_b.append(b[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and F[i][j] == F[i - 1][j] + hueco:
            al_a.append(a[i - 1])
            al_b.append("-")
            i -= 1
        else:
            al_a.append("-")
            al_b.append(b[j - 1])
            j -= 1
    al_a.reverse()
    al_b.reverse()
    return F[n][m], F, al_a, al_b


def similitud_needleman_wunsch(a, b):
    """Porcentaje de identidad del alineamiento: coincidencias / longitud alineada."""
    if len(a) == 0 and len(b) == 0:
        return 1.0
    puntaje, F, al_a, al_b = needleman_wunsch(a, b)
    coincidencias = 0
    for k in range(len(al_a)):
        if al_a[k] == al_b[k]:
            coincidencias += 1
    return coincidencias / len(al_a)


# ---------------------------------------------------------------------------
# 3. TF-IDF + coseno
# ---------------------------------------------------------------------------
def frecuencias(terminos):
    tf = {}
    for t in terminos:
        if t in tf:
            tf[t] += 1
        else:
            tf[t] = 1
    return tf


def calcular_idf(documentos):
    """documentos: lista de listas de terminos. idf(t) = ln(N / df(t))"""
    N = len(documentos)
    df = {}
    for doc in documentos:
        vistos = set(doc)
        for t in vistos:
            df[t] = df.get(t, 0) + 1
    idf = {}
    for t in df:
        idf[t] = math.log(N / df[t])
    return idf, df


def vector_tfidf(terminos, idf):
    """tf normalizado (frecuencia / total de terminos) multiplicado por idf."""
    tf = frecuencias(terminos)
    total = len(terminos)
    vec = {}
    for t in tf:
        vec[t] = (tf[t] / total) * idf.get(t, 0.0)
    return vec


def coseno_dispersos(v1, v2):
    """Coseno entre dos vectores guardados como diccionario {termino: peso}."""
    producto = 0.0
    for t in v1:
        if t in v2:
            producto += v1[t] * v2[t]
    norma1 = 0.0
    for t in v1:
        norma1 += v1[t] * v1[t]
    norma2 = 0.0
    for t in v2:
        norma2 += v2[t] * v2[t]
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return producto / (math.sqrt(norma1) * math.sqrt(norma2))


def similitud_coseno_tfidf(terminos_a, terminos_b, idf):
    return coseno_dispersos(vector_tfidf(terminos_a, idf), vector_tfidf(terminos_b, idf))


# ---------------------------------------------------------------------------
# 4. Jaccard
# ---------------------------------------------------------------------------
def jaccard(terminos_a, terminos_b):
    A = set(terminos_a)
    B = set(terminos_b)
    interseccion = 0
    for t in A:
        if t in B:
            interseccion += 1
    union = len(A) + len(B) - interseccion
    if union == 0:
        return 1.0
    return interseccion / union
