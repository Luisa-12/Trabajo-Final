# Algoritmo Knuth-Morris-Pratt (KMP)
# Preprocesamiento: tabla de prefijos (funcion de fallo) en O(m)
# Busqueda: O(n), el indice sobre el texto nunca retrocede.


def tabla_prefijos(patron):
    """pi[q] = longitud del prefijo propio mas largo de patron[0..q] que tambien
    es sufijo de patron[0..q]."""
    m = len(patron)
    pi = [0] * m
    k = 0
    for q in range(1, m):
        while k > 0 and patron[k] != patron[q]:
            k = pi[k - 1]       # no coincide: vuelvo al prefijo más corto que todavía sirve
        if patron[k] == patron[q]:
            k += 1               # coincide: el prefijo que se repite crece en una letra
        pi[q] = k
    return pi


def buscar(texto, patron, traza=False):
    """Devuelve (posiciones, comparaciones, pasos). pasos solo se llena si traza=True."""
    n = len(texto)
    m = len(patron)
    posiciones = []
    pasos = []
    if m == 0 or m > n:
        return posiciones, 0, pasos
    pi = tabla_prefijos(patron)
    comparaciones = 0
    q = 0  # cantidad de caracteres del patron que ya coinciden
    for i in range(n):
        while q > 0 and patron[q] != texto[i]:
            comparaciones += 1
            if traza:
                pasos.append({"i": i, "q": q, "accion": "fallo: '%s' != '%s', q = pi[%d] = %d" % (patron[q], texto[i], q - 1, pi[q - 1])})
            q = pi[q - 1]  # fallo en el texto: uso la tabla en vez de empezar de cero
        comparaciones += 1
        if patron[q] == texto[i]:
            q += 1
            if traza:
                pasos.append({"i": i, "q": q, "accion": "coincide '%s', q = %d" % (texto[i], q)})
        elif traza:
            pasos.append({"i": i, "q": q, "accion": "no coincide '%s' != '%s', avanza i" % (patron[q], texto[i])})
        if q == m:
            posiciones.append(i - m + 1)  # encontré el patrón completo; guardo dónde empieza
            if traza:
                pasos.append({"i": i, "q": q, "accion": "OCURRENCIA en %d, q = pi[%d] = %d" % (i - m + 1, m - 1, pi[m - 1])})
            q = pi[q - 1]
    return posiciones, comparaciones, pasos
