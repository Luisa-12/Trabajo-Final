# Dibujo del dendrograma a partir de la lista de uniones.
# Solo se usan lineas de matplotlib; la posicion de cada nodo se calcula a mano:
#   x de una hoja = su posicion en el orden de dibujo
#   x de un grupo = punto medio entre las x de sus dos hijos
#   y de un grupo = distancia a la que se unieron

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clustering.jerarquico import hojas_de

COLORES = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d"]


def posiciones(uniones, n):
    orden = hojas_de(n + len(uniones) - 1, uniones, n)
    x = {}
    y = {}
    for pos in range(len(orden)):
        x[orden[pos]] = pos
        y[orden[pos]] = 0.0
    for k in range(len(uniones)):
        a, b, d, t = uniones[k]
        x[n + k] = (x[a] + x[b]) / 2
        y[n + k] = d
    return orden, x, y


def color_por_grupo(uniones, n, k):
    """Asigna un color a cada nodo segun el grupo al cortar en k grupos.
    Las k-1 uniones mas altas se dibujan en gris."""
    color = {}
    limite = n - k  # las uniones con indice < limite quedan dentro de un grupo
    raices = []
    for paso in range(limite, len(uniones)):
        a, b, d, t = uniones[paso]
        for hijo in (a, b):
            if hijo < n or hijo - n < limite:
                raices.append(hijo)
    for idx in range(len(raices)):
        c = COLORES[idx % len(COLORES)]
        pila = [raices[idx]]
        while pila:
            g = pila.pop()
            color[g] = c
            if g >= n:
                a, b, d, t = uniones[g - n]
                pila.append(a)
                pila.append(b)
    return color


def dibujar(uniones, etiquetas, titulo, k_colores=3, ruta=None):
    n = len(etiquetas)
    orden, x, y = posiciones(uniones, n)
    color = color_por_grupo(uniones, n, k_colores)
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(len(uniones)):
        a, b, d, t = uniones[k]
        c = color.get(n + k, "#6b7280")
        # dos lineas verticales desde cada hijo y una horizontal que las une
        ax.plot([y[a], d], [x[a], x[a]], color=c, linewidth=1.6)
        ax.plot([y[b], d], [x[b], x[b]], color=c, linewidth=1.6)
        ax.plot([d, d], [x[a], x[b]], color=c, linewidth=1.6)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([etiquetas[i] for i in orden], fontsize=8)
    ax.set_xlabel("Distancia (1 - similitud)")
    ax.set_title(titulo)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    if ruta is not None:
        fig.savefig(ruta, dpi=150)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
