# Carga del corpus (titulo, autores y abstract de los 20 articulos)

import os
import json

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_CORPUS = os.path.join(BASE, "datos", "corpus.json")


def cargar_corpus():
    with open(RUTA_CORPUS, encoding="utf-8") as f:
        return json.load(f)


def buscar_articulo(corpus, id_articulo):
    for art in corpus:
        if art["id"] == id_articulo:
            return art
    return None


def autores_en_comun(art1, art2):
    """Cruce de metadatos: autores que aparecen en ambos articulos."""
    comunes = []
    for a in art1["autores"]:
        if a in art2["autores"]:
            comunes.append(a)
    return comunes


def construir_corpus():
    """Une la extraccion automatica de los PDF con los metadatos verificados."""
    from extraccion.extractor_pdf import extraer_todo
    extraidos = extraer_todo(os.path.join(BASE, "datos", "pdfs"))
    with open(os.path.join(BASE, "datos", "metadatos.json"), encoding="utf-8") as f:
        metadatos = json.load(f)
    corpus = []
    for meta in metadatos:
        for ex in extraidos:
            if ex["id"] == meta["id"]:
                art = dict(meta)
                art["archivo"] = ex["archivo"]
                art["abstract"] = ex["abstract"]
                corpus.append(art)
    with open(RUTA_CORPUS, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    return corpus


if __name__ == "__main__":
    c = construir_corpus()
    print("Corpus construido con", len(c), "articulos")
