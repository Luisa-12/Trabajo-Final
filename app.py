# Aplicacion web del proyecto: Analisis de algoritmos en el contexto de la bibliometria
# Ejecutar localmente:  python app.py   ->  http://127.0.0.1:5000

import os
import base64
from flask import Flask, render_template, request

from corpus import cargar_corpus, buscar_articulo, autores_en_comun
from preprocesamiento import preprocesar, tokenizar
from similitud import clasicos
from similitud.comparador import preparar_idf, comparar_varios, NOMBRES
from clustering import jerarquico, dendrograma
from busqueda import kmp, boyer_moore, rabin_karp
from busqueda.buscador import buscar_en_corpus, a_minusculas
from busqueda.rendimiento import comparar as comparar_rendimiento

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

CORPUS = cargar_corpus()
IDF = preparar_idf(CORPUS)
_cache_clustering = {}


def imagen_base64(png):
    return base64.b64encode(png).decode("ascii")


@app.route("/")
def inicio():
    return render_template("inicio.html", corpus=CORPUS)


@app.route("/articulo/<int:id_articulo>")
def articulo(id_articulo):
    art = buscar_articulo(CORPUS, id_articulo)
    return render_template("articulo.html", art=art, terminos=preprocesar(art["abstract"]))


# ----------------------------- Requerimiento 1 -----------------------------
@app.route("/similitud", methods=["GET", "POST"])
def similitud():
    seleccion = []
    pares = []
    matrices = None
    n_palabras = 8
    if request.method == "POST":
        seleccion = [int(x) for x in request.form.getlist("articulos")]
        n_palabras = int(request.form.get("n_palabras", 8))
        n_palabras = max(3, min(n_palabras, 15))
        if len(seleccion) >= 2:
            articulos = [buscar_articulo(CORPUS, i) for i in seleccion]
            for a, b, res in comparar_varios(articulos, IDF):
                pares.append({"a": a, "b": b, "res": res, "autores_comunes": autores_en_comun(a, b)})
            # demostracion: matrices de programacion dinamica con las primeras palabras
            ta = preprocesar(articulos[0]["abstract"])[:n_palabras]
            tb = preprocesar(articulos[1]["abstract"])[:n_palabras]
            dist, D = clasicos.levenshtein(ta, tb)
            punt, F, al_a, al_b = clasicos.needleman_wunsch(ta, tb)
            matrices = {"a": ta, "b": tb, "lev": D, "dist": dist, "nw": F, "puntaje": punt,
                        "al_a": al_a, "al_b": al_b, "id_a": articulos[0]["id"], "id_b": articulos[1]["id"]}
    return render_template("similitud.html", corpus=CORPUS, seleccion=seleccion, pares=pares,
                           nombres=NOMBRES, matrices=matrices, n_palabras=n_palabras)


# ----------------------------- Requerimiento 2 -----------------------------
def calcular_clustering(representacion):
    if representacion in _cache_clustering:
        return _cache_clustering[representacion]
    if representacion == "minilm":
        D = jerarquico.matriz_distancias_minilm(CORPUS)
        nombre = "Sentence-BERT"
    else:
        D = jerarquico.matriz_distancias_tfidf(CORPUS)
        nombre = "TF-IDF + coseno"
    etiquetas = ["%d. %s" % (a["id"], a["titulo"][:48]) for a in CORPUS]
    resultados = []
    for metodo, titulo in [("single", "Single linkage"), ("complete", "Complete linkage"), ("average", "Average linkage (UPGMA)")]:
        uniones = jerarquico.aglomerativo(D, metodo)
        evaluacion = jerarquico.evaluar(D, uniones)
        png = dendrograma.dibujar(uniones, etiquetas, "%s - %s" % (titulo, nombre), k_colores=3)
        grupos = jerarquico.cortar_arbol(uniones, len(CORPUS), 3)
        miembros = {}
        for i in range(len(CORPUS)):
            miembros.setdefault(grupos[i], []).append(CORPUS[i]["id"])
        resultados.append({"metodo": metodo, "titulo": titulo, "imagen": imagen_base64(png),
                           "eval": evaluacion, "grupos": list(miembros.values())})
    mejor = max(resultados, key=lambda r: r["eval"]["ccc"])
    _cache_clustering[representacion] = (resultados, mejor["titulo"], nombre)
    return _cache_clustering[representacion]


@app.route("/clustering")
def clustering():
    representacion = request.args.get("representacion", "tfidf")
    resultados, mejor, nombre = calcular_clustering(representacion)
    return render_template("clustering.html", resultados=resultados, mejor=mejor,
                           representacion=representacion, nombre=nombre)


# ----------------------------- Requerimiento 3 -----------------------------
@app.route("/busqueda", methods=["GET", "POST"])
def busqueda():
    patron = ""
    resumen = None
    coincidencias = []
    demo = None
    mayusculas = False
    if request.method == "POST":
        patron = request.form.get("patron", "")
        mayusculas = request.form.get("mayusculas") == "si"
        if patron.strip() != "":
            resumen, coincidencias = buscar_en_corpus(CORPUS, patron, mayusculas)
            p = patron if mayusculas else a_minusculas(patron)
            if len(p) <= 30:
                demo = {
                    "pi": kmp.tabla_prefijos(p),
                    "malo": sorted(boyer_moore.tabla_caracter_malo(p).items(), key=lambda x: x[1]),
                    "sufijo": boyer_moore.tabla_sufijo_bueno(p),
                    "hash": rabin_karp.hash_cadena(p),
                    "patron": list(p),
                }
    return render_template("busqueda.html", patron=patron, resumen=resumen, coincidencias=coincidencias,
                           demo=demo, mayusculas=mayusculas)


@app.route("/traza", methods=["GET", "POST"])
def traza():
    """Ejecucion paso a paso de los tres algoritmos sobre un fragmento corto."""
    texto = request.form.get("texto", "generative artificial intelligence transforms assessment in higher education")
    patron = request.form.get("patron", "assessment")
    primo = int(request.form.get("primo", 101))
    texto = texto[:400]
    pk = kmp.buscar(texto, patron, traza=True)
    pb = boyer_moore.buscar(texto, patron, traza=True)
    pr = rabin_karp.buscar(texto, patron, q=primo, traza=True)
    return render_template("traza.html", texto=texto, patron=patron, primo=primo,
                           kmp=pk, bm=pb, rk=pr, pi=kmp.tabla_prefijos(patron) if patron else [],
                           malo=sorted(boyer_moore.tabla_caracter_malo(patron).items(), key=lambda x: x[1]),
                           sufijo=boyer_moore.tabla_sufijo_bueno(patron) if patron else [])


@app.route("/rendimiento", methods=["GET", "POST"])
def rendimiento():
    patron = request.form.get("patron", "artificial intelligence")
    repetir = int(request.form.get("repetir", 1))
    repetir = max(1, min(repetir, 10))
    texto = ""
    for art in CORPUS:
        texto = texto + a_minusculas(art["abstract"]) + "\n"
    texto = texto * repetir
    filas = comparar_rendimiento(texto, a_minusculas(patron)) if patron else []
    return render_template("rendimiento.html", patron=patron, repetir=repetir, n=len(texto), filas=filas)


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
