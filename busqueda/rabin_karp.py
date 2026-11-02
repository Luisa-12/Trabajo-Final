# Algoritmo Rabin-Karp
# Convierte el patron y cada ventana del texto en un numero (hash) usando
# aritmetica modular. La ventana siguiente se calcula a partir de la anterior
# en O(1) (rolling hash):
#   h(s+1) = (d * (h(s) - T[s] * d^(m-1)) + T[s+m]) mod q
# Si los hash coinciden se verifica caracter a caracter, porque dos cadenas
# distintas pueden tener el mismo hash (colision o "falso positivo").

BASE = 256           # d: tamano del alfabeto (codigos de caracter)
PRIMO = 1000000007   # q: modulo grande -> pocas colisiones


def hash_cadena(cadena, d=BASE, q=PRIMO):
    """Metodo de Horner: h = (...((c0*d + c1)*d + c2)...) mod q"""
    h = 0
    for c in cadena:
        h = (h * d + ord(c)) % q
    return h


def buscar(texto, patron, d=BASE, q=PRIMO, traza=False):
    """Devuelve (posiciones, comparaciones, pasos, colisiones)."""
    n = len(texto)
    m = len(patron)
    posiciones = []
    pasos = []
    colisiones = 0
    if m == 0 or m > n:
        return posiciones, 0, pasos, 0
    # h = d^(m-1) mod q, para quitar el primer caracter de la ventana
    h = 1
    for _ in range(m - 1):
        h = (h * d) % q
    hp = 0
    ht = 0
    for i in range(m):
        hp = (d * hp + ord(patron[i])) % q
        ht = (d * ht + ord(texto[i])) % q
    comparaciones = 0
    for s in range(n - m + 1):
        if hp == ht:
            # verificacion caracter a caracter
            iguales = True
            for j in range(m):
                comparaciones += 1
                if texto[s + j] != patron[j]:
                    iguales = False
                    break
            if iguales:
                posiciones.append(s)
                if traza:
                    pasos.append({"s": s, "hash": ht, "accion": "hash igual y verificado: OCURRENCIA"})
            else:
                colisiones += 1
                if traza:
                    pasos.append({"s": s, "hash": ht, "accion": "hash igual pero texto distinto: COLISION"})
        elif traza:
            pasos.append({"s": s, "hash": ht, "accion": "hash distinto"})
        if s < n - m:
            ht = (d * (ht - ord(texto[s]) * h) + ord(texto[s + m])) % q
    return posiciones, comparaciones, pasos, colisiones
