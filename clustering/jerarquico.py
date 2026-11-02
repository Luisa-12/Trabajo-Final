# Agrupamiento jerarquico aglomerativo implementado desde cero.
# Criterios: single linkage, complete linkage y average linkage (UPGMA).
#
# Idea: se empieza con cada abstract como un grupo. En cada paso se buscan los
# dos grupos mas cercanos, se unen y se recalcula la distancia del nuevo grupo
# con los demas usando la formula de Lance-Williams:
#   single:   d(k, i U j) = min(d(k,i), d(k,j))
#   complete: d(k, i U j) = max(d(k,i), d(k,j))
#   average:  d(k, i U j) = (|i| d(k,i) + |j| d(k,j)) / (|i| + |j|)

import math
from preprocesamiento import preprocesar
from similitud import clasicos, ia


# ----------------------------- matriz de distancias -----------------------------
def matriz_distancias_tfidf(corpus):
    documentos = []
    for art in corpus:
        documentos.append(preprocesar(art["abstract"]))
    idf, df = clasicos.calcular_idf(documentos)
    vectores = []
    for doc in documentos:
        vectores.append(clasicos.vector_tfidf(doc, idf))
    n = len(corpus)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - clasicos.coseno_dispersos(vectores[i], vectores[j])
            D[i][j] = d
            D[j][i] = d
    return D


def matriz_distancias_minilm(corpus):
    n = len(corpus)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s, _ = ia.similitud_minilm(corpus[i]["id"], corpus[j]["id"])
            D[i][j] = 1 - s
            D[j][i] = 1 - s
    return D


# ----------------------------- algoritmo aglomerativo -----------------------------
def nueva_distancia(metodo, d_ki, d_kj, tam_i, tam_j):
    if metodo == "single":
        return min(d_ki, d_kj)
    if metodo == "complete":
        return max(d_ki, d_kj)
    if metodo == "average":
        return (tam_i * d_ki + tam_j * d_kj) / (tam_i + tam_j)
    raise ValueError("metodo desconocido: " + metodo)


def aglomerativo(D, metodo):
    """D: matriz de distancias n x n (lista de listas).
    Devuelve la lista de uniones [(grupo_a, grupo_b, distancia, tamano)], con la
    misma convencion de scipy: los grupos 0..n-1 son las hojas y el grupo creado
    en el paso k recibe el numero n + k."""
    n = len(D)
    # distancias entre grupos activos, guardadas en un diccionario {(a, b): d} con a < b
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i][j]
    activos = list(range(n))
    tamano = {}
    for i in range(n):
        tamano[i] = 1
    uniones = []
    siguiente = n
    while len(activos) > 1:
        # 1) buscar el par de grupos mas cercano (recorrido completo, O(g^2))
        mejor = None
        mejor_d = math.inf
        for x in range(len(activos)):
            for y in range(x + 1, len(activos)):
                a = activos[x]
                b = activos[y]
                d = dist[(min(a, b), max(a, b))]
                if d < mejor_d:
                    mejor_d = d
                    mejor = (a, b)
        a, b = mejor
        # 2) crear el nuevo grupo y calcular su distancia a los demas
        nuevo = siguiente
        siguiente += 1
        tamano[nuevo] = tamano[a] + tamano[b]
        for k in activos:
            if k != a and k != b:
                d_ka = dist[(min(k, a), max(k, a))]
                d_kb = dist[(min(k, b), max(k, b))]
                dist[(k, nuevo)] = nueva_distancia(metodo, d_ka, d_kb, tamano[a], tamano[b])
        activos.remove(a)
        activos.remove(b)
        activos.append(nuevo)
        uniones.append((a, b, mejor_d, tamano[nuevo]))
    return uniones


# ----------------------------- utilidades sobre el arbol -----------------------------
def hojas_de(grupo, uniones, n):
    """Lista de hojas (indices de abstracts) que contiene un grupo, en orden de dibujo."""
    if grupo < n:
        return [grupo]
    a, b, d, t = uniones[grupo - n]
    return hojas_de(a, uniones, n) + hojas_de(b, uniones, n)


def distancias_cofeneticas(uniones, n):
    """C[i][j] = altura del dendrograma a la que i y j quedan en el mismo grupo."""
    C = [[0.0] * n for _ in range(n)]
    for k in range(len(uniones)):
        a, b, d, t = uniones[k]
        ha = hojas_de(a, uniones, n)
        hb = hojas_de(b, uniones, n)
        for i in ha:
            for j in hb:
                C[i][j] = d
                C[j][i] = d
    return C


def cortar_arbol(uniones, n, k):
    """Etiquetas de grupo al cortar el arbol en k grupos (se deshacen las k-1 ultimas uniones)."""
    etiquetas = list(range(n))
    padre = {}
    for i in range(n):
        padre[i] = i
    # aplicar las primeras n-k uniones
    for paso in range(n - k):
        a, b, d, t = uniones[paso]
        nuevo = n + paso
        for hoja in hojas_de(nuevo, uniones, n):
            padre[hoja] = nuevo
    # renumerar 0..k-1
    mapa = {}
    for i in range(n):
        g = padre[i]
        if g not in mapa:
            mapa[g] = len(mapa)
        etiquetas[i] = mapa[g]
    return etiquetas


# ----------------------------- metricas de evaluacion interna -----------------------------
def correlacion_cofenetica(D, uniones):
    """Coeficiente de correlacion cofenetica (CCC): correlacion de Pearson entre
    las distancias originales y las distancias cofeneticas. Mientras mas cerca de 1,
    mejor representa el dendrograma las distancias reales."""
    n = len(D)
    C = distancias_cofeneticas(uniones, n)
    x = []
    y = []
    for i in range(n):
        for j in range(i + 1, n):
            x.append(D[i][j])
            y.append(C[i][j])
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = 0.0
    sx = 0.0
    sy = 0.0
    for k in range(len(x)):
        num += (x[k] - mx) * (y[k] - my)
        sx += (x[k] - mx) ** 2
        sy += (y[k] - my) ** 2
    return num / math.sqrt(sx * sy)


def silueta(D, etiquetas):
    """Coeficiente de silueta promedio. Para cada punto i:
    a(i) = distancia promedio a los de su grupo, b(i) = menor distancia promedio a otro grupo,
    s(i) = (b - a) / max(a, b). Los puntos solos en su grupo tienen s(i) = 0."""
    n = len(D)
    grupos = set(etiquetas)
    total = 0.0
    for i in range(n):
        mismos = [j for j in range(n) if etiquetas[j] == etiquetas[i] and j != i]
        if len(mismos) == 0:
            continue
        a = sum(D[i][j] for j in mismos) / len(mismos)
        b = math.inf
        for g in grupos:
            if g == etiquetas[i]:
                continue
            otros = [j for j in range(n) if etiquetas[j] == g]
            prom = sum(D[i][j] for j in otros) / len(otros)
            if prom < b:
                b = prom
        total += (b - a) / max(a, b)
    return total / n


def evaluar(D, uniones, ks=(2, 3, 4, 5, 6)):
    n = len(D)
    resultado = {"ccc": correlacion_cofenetica(D, uniones), "silueta": {}, "tamanos": {}}
    for k in ks:
        etiquetas = cortar_arbol(uniones, n, k)
        resultado["silueta"][k] = silueta(D, etiquetas)
        conteo = [0] * k
        for e in etiquetas:
            conteo[e] += 1
        resultado["tamanos"][k] = sorted(conteo, reverse=True)
    return resultado
