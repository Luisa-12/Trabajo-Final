# Busqueda por fuerza bruta. No es uno de los algoritmos pedidos; se usa
# como linea base en las pruebas y en la comparacion de rendimiento. O(n*m)


def buscar(texto, patron):
    n = len(texto)
    m = len(patron)
    posiciones = []
    comparaciones = 0
    if m == 0 or m > n:
        return posiciones, 0
    for s in range(n - m + 1):
        j = 0
        while j < m:
            comparaciones += 1
            if texto[s + j] != patron[j]:
                break
            j += 1
        if j == m:
            posiciones.append(s)
    return posiciones, comparaciones
