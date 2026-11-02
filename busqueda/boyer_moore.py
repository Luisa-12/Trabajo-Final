# Algoritmo Boyer-Moore
# Compara el patron de DERECHA a IZQUIERDA y, cuando falla, desplaza usando
# el maximo entre dos heuristicas:
#   - Regla del caracter malo: alinear el caracter del texto que fallo con su
#     ultima aparicion en el patron.
#   - Regla del sufijo bueno: alinear el sufijo que si coincidio con otra
#     aparicion de ese sufijo en el patron (o con un prefijo que lo sea).


def tabla_caracter_malo(patron):
    """ultima[c] = ultima posicion del caracter c en el patron."""
    ultima = {}
    for i in range(len(patron)):
        ultima[patron[i]] = i
    return ultima


def tabla_sufijo_bueno(patron):
    """desplazamiento[j] = cuanto mover el patron si falla en la posicion j-1
    despues de haber coincidido patron[j..m-1]. Se construye con el arreglo
    'borde' (version fuerte de la regla, Cormen / Charras-Lecroq)."""
    m = len(patron)
    desplazamiento = [0] * (m + 1)
    borde = [0] * (m + 1)
    # caso 1: el sufijo coincidente aparece en otra parte del patron
    i = m
    j = m + 1
    borde[i] = j
    while i > 0:
        while j <= m and patron[i - 1] != patron[j - 1]:
            if desplazamiento[j] == 0:
                desplazamiento[j] = j - i
            j = borde[j]
        i -= 1
        j -= 1
        borde[i] = j
    # caso 2: solo una parte del sufijo coincide con un prefijo del patron
    j = borde[0]
    for i in range(m + 1):
        if desplazamiento[i] == 0:
            desplazamiento[i] = j
        if i == j:
            j = borde[j]
    return desplazamiento


def buscar(texto, patron, traza=False):
    """Devuelve (posiciones, comparaciones, pasos)."""
    n = len(texto)
    m = len(patron)
    posiciones = []
    pasos = []
    if m == 0 or m > n:
        return posiciones, 0, pasos
    ultima = tabla_caracter_malo(patron)
    sufijo = tabla_sufijo_bueno(patron)
    comparaciones = 0
    s = 0  # desplazamiento actual del patron sobre el texto
    while s <= n - m:
        j = m - 1
        while j >= 0:
            comparaciones += 1
            if patron[j] != texto[s + j]:
                break
            j -= 1
        if j < 0:
            posiciones.append(s)
            salto = sufijo[0]
            if traza:
                pasos.append({"s": s, "j": -1, "accion": "OCURRENCIA en %d; sufijo bueno completo -> desplaza %d" % (s, salto)})
            s += salto
        else:
            c = texto[s + j]
            salto_malo = j - ultima.get(c, -1)
            salto_bueno = sufijo[j + 1]
            salto = max(salto_malo, salto_bueno, 1)
            if traza:
                pasos.append({"s": s, "j": j, "accion": "falla en j=%d ('%s' vs '%s'): caracter malo=%d, sufijo bueno=%d -> desplaza %d"
                              % (j, patron[j], c, salto_malo, salto_bueno, salto)})
            s += salto
    return posiciones, comparaciones, pasos
