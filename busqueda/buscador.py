# Busca un patron en los abstracts del corpus con los tres algoritmos y
# junta los resultados con el titulo y los autores de cada articulo.

import time
from busqueda import kmp, boyer_moore, rabin_karp, ingenuo


def a_minusculas(texto):
    """Conversion manual a minusculas (solo letras A-Z y vocales con tilde)."""
    resultado = []
    for c in texto:
        if "A" <= c <= "Z":
            resultado.append(chr(ord(c) + 32))
        else:
            resultado.append(c)
    return "".join(resultado)


def ejecutar(algoritmo, texto, patron):
    inicio = time.perf_counter()
    if algoritmo == "kmp":
        pos, comp, _ = kmp.buscar(texto, patron)
    elif algoritmo == "boyer_moore":
        pos, comp, _ = boyer_moore.buscar(texto, patron)
    elif algoritmo == "rabin_karp":
        pos, comp, _, _ = rabin_karp.buscar(texto, patron)
    else:
        pos, comp = ingenuo.buscar(texto, patron)
    return pos, comp, (time.perf_counter() - inicio) * 1000


def contexto(texto, posicion, largo, ancho=60):
    ini = max(0, posicion - ancho)
    fin = min(len(texto), posicion + largo + ancho)
    return texto[ini:posicion], texto[posicion:posicion + largo], texto[posicion + largo:fin]


def buscar_en_corpus(corpus, patron, distinguir_mayusculas=False):
    """Devuelve el resumen por algoritmo y las coincidencias por articulo."""
    if not distinguir_mayusculas:
        patron_b = a_minusculas(patron)
    else:
        patron_b = patron
    resumen = {}
    for alg in ["kmp", "boyer_moore", "rabin_karp"]:
        resumen[alg] = {"ocurrencias": 0, "comparaciones": 0, "tiempo_ms": 0.0}
    por_articulo = []
    for art in corpus:
        texto = art["abstract"]
        texto_b = texto if distinguir_mayusculas else a_minusculas(texto)
        posiciones_ref = None
        for alg in ["kmp", "boyer_moore", "rabin_karp"]:
            pos, comp, t = ejecutar(alg, texto_b, patron_b)
            resumen[alg]["ocurrencias"] += len(pos)
            resumen[alg]["comparaciones"] += comp
            resumen[alg]["tiempo_ms"] += t
            if posiciones_ref is None:
                posiciones_ref = pos
        if posiciones_ref:
            fragmentos = []
            for p in posiciones_ref:
                fragmentos.append(contexto(texto, p, len(patron_b)))
            por_articulo.append({"articulo": art, "posiciones": posiciones_ref, "fragmentos": fragmentos})
    return resumen, por_articulo
