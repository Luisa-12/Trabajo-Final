# Medicion de rendimiento: separa el tiempo de PREPROCESAMIENTO del patron
# y el tiempo de BUSQUEDA sobre el texto, y cuenta comparaciones de caracteres.
# Cada medicion se repite varias veces y se reportan media, desviacion estandar,
# intervalo de confianza del 95 % y el minimo.

import math
import time
from busqueda import kmp, boyer_moore, rabin_karp, ingenuo

REPETICIONES = 5

# valores t de Student (dos colas, 95 %) para algunos grados de libertad
T_STUDENT = {4: 2.776, 9: 2.262, 19: 2.093, 29: 2.045}


def estadisticas(tiempos):
    n = len(tiempos)
    media = sum(tiempos) / n
    if n > 1:
        var = sum((t - media) ** 2 for t in tiempos) / (n - 1)
    else:
        var = 0.0
    de = math.sqrt(var)
    t = T_STUDENT.get(n - 1, 1.96)
    margen = t * de / math.sqrt(n) if n > 1 else 0.0
    return {"media": media, "de": de, "ic95": margen, "minimo": min(tiempos), "n": n}


def medir(funcion, *args, repeticiones=None):
    """Ejecuta la funcion varias veces. Devuelve (estadisticas en ms, resultado)."""
    if repeticiones is None:
        repeticiones = REPETICIONES
    tiempos = []
    resultado = None
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        resultado = funcion(*args)
        tiempos.append((time.perf_counter() - inicio) * 1000)
    return estadisticas(tiempos), resultado


def preprocesar_rk(patron):
    h = 1
    for _ in range(len(patron) - 1):
        h = (h * rabin_karp.BASE) % rabin_karp.PRIMO
    return h, rabin_karp.hash_cadena(patron)


def preprocesar_bm(patron):
    return boyer_moore.tabla_caracter_malo(patron), boyer_moore.tabla_sufijo_bueno(patron)


def comparar(texto, patron, repeticiones=None):
    filas = []
    casos = [
        ("KMP", kmp.tabla_prefijos, kmp.buscar),
        ("Boyer-Moore", preprocesar_bm, boyer_moore.buscar),
        ("Rabin-Karp", preprocesar_rk, rabin_karp.buscar),
    ]
    for nombre, prepro, busq in casos:
        e_pre, _ = medir(prepro, patron, repeticiones=repeticiones)
        e_tot, res = medir(busq, texto, patron, repeticiones=repeticiones)
        filas.append({
            "algoritmo": nombre,
            "preprocesamiento_ms": e_pre["media"],
            "busqueda_ms": max(e_tot["media"] - e_pre["media"], 0.0),
            "total_ms": e_tot["media"],
            "total_de": e_tot["de"],
            "total_ic95": e_tot["ic95"],
            "total_min": e_tot["minimo"],
            "comparaciones": res[1],
            "ocurrencias": len(res[0]),
            "colisiones": res[3] if nombre == "Rabin-Karp" else None,
        })
    e_tot, res = medir(ingenuo.buscar, texto, patron, repeticiones=repeticiones)
    filas.append({"algoritmo": "Fuerza bruta (referencia)", "preprocesamiento_ms": 0.0, "busqueda_ms": e_tot["media"],
                  "total_ms": e_tot["media"], "total_de": e_tot["de"], "total_ic95": e_tot["ic95"],
                  "total_min": e_tot["minimo"], "comparaciones": res[1], "ocurrencias": len(res[0]), "colisiones": None})
    return filas


def regresion_lineal(xs, ys):
    """Minimos cuadrados y = a + b x. Devuelve (a, b, R^2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot
