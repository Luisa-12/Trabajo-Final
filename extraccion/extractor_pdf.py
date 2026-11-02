# Extraccion del abstract, titulo y autores de los PDF del corpus.
# Se lee la primera y segunda pagina con pypdf y se aplican reglas
# sencillas para encontrar donde empieza y donde termina el abstract.

import os
import re
import json
from pypdf import PdfReader

# Palabras que marcan el inicio del abstract segun la editorial
MARCAS_INICIO = ["A B S T R A C T", "ABSTRACT", "Abstract:", "Abstract"]

# Palabras que marcan el final del abstract
MARCAS_FIN = ["Keywords", "KEYWORDS", "Key words", "1. Introduction", "1 | INT",
              "Introduction", "Citation:", "Paper type", "* Fan Ouyang",
              "1. Artificial", "Artificial Intelligence in STEM Education",
              "Since Alan Turing"]


def leer_texto_pdf(ruta, paginas=2):
    lector = PdfReader(ruta)
    texto = ""
    for i in range(min(paginas, len(lector.pages))):
        texto = texto + lector.pages[i].extract_text() + "\n"
    return texto


CORRECCIONES = {"de fi-nitions": "definitions", "stakeholders-like": "stakeholders - like"}


def cargar_diccionario():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diccionario_ingles.txt")
    with open(ruta, encoding="utf-8") as f:
        return set(f.read().split())


DICCIONARIO = cargar_diccionario()


def unir_si_es_palabra(izq, der, separador):
    """Une dos fragmentos ("Arti" + "ficial", "multidis" + "ciplinary") solo si el
    resultado es una palabra del diccionario. Si no, se deja el separador."""
    unida = (izq + der).lower()
    if unida in DICCIONARIO and izq.lower() != "a":
        return izq + der
    return izq + separador + der


def limpiar_texto(texto):
    # guiones largos y comillas tipograficas
    reemplazos = {"­": "", "’": "'", "“": '"', "”": '"', "–": " - ", "—": " - "}
    for viejo, nuevo in reemplazos.items():
        texto = texto.replace(viejo, nuevo)
    # ligaduras: pypdf a veces deja un espacio antes de ellas ("Arti ﬁcial", "the ﬁeld")
    texto = texto.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("ﬃ", "ffi")
    texto = re.sub(r"([A-Za-z]+) ((?:fi|fl|ff)[a-z]+)",
                   lambda m: unir_si_es_palabra(m.group(1), m.group(2), " "), texto)
    # o un espacio despues de la ligadura ("deﬁ nitions")
    texto = re.sub(r"([A-Za-z]*(?:fi|fl|ff)) ([a-z]+)",
                   lambda m: unir_si_es_palabra(m.group(1), m.group(2), " "), texto)
    # palabras partidas al final de linea: "multidis-\nciplinary" -> se une,
    # "AI-\ndriven" -> se deja con guion porque "aidriven" no es palabra
    texto = re.sub(r"([A-Za-z]+) ?-[ \t]*\n\s*([a-z]+)",
                   lambda m: unir_si_es_palabra(m.group(1), m.group(2), "-"), texto)
    texto = re.sub(r"(\w)-  (\w)", r"\1-\2", texto)   # "well-  defined"
    # casos puntuales revisados a mano que las reglas no resuelven
    for viejo, nuevo in CORRECCIONES.items():
        texto = texto.replace(viejo, nuevo)
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"(\w) ' ", r"\1' ", texto)   # "learners ' answers"
    texto = re.sub(r" ([.,;])", r"\1", texto)     # espacio antes de punto o coma
    return texto.strip()


def buscar_inicio(lineas):
    for i in range(len(lineas)):
        linea = lineas[i].strip()
        for marca in MARCAS_INICIO:
            if linea.startswith(marca) or linea.endswith(marca):
                return i, marca
    return -1, None


def es_linea_de_palabras_clave(linea):
    # las palabras clave de Elsevier aparecen como lineas cortas
    return len(linea.strip()) < 48


def extraer_abstract(texto):
    lineas = texto.split("\n")
    inicio, marca = buscar_inicio(lineas)
    if inicio == -1:
        return ""
    # lo que queda en la misma linea despues de la marca
    primera = lineas[inicio].strip()
    resto = primera[primera.find(marca) + len(marca):].strip() if primera.startswith(marca) else ""
    j = inicio + 1
    # si justo despues viene el bloque de "Keywords:" (Elsevier) se salta
    if j < len(lineas) and lineas[j].strip().startswith("Keywords"):
        j = j + 1
        while j < len(lineas) and es_linea_de_palabras_clave(lineas[j]):
            j = j + 1
    partes = [resto] if resto else []
    while j < len(lineas):
        linea = lineas[j].strip()
        fin = False
        for m in MARCAS_FIN:
            if linea.startswith(m):
                fin = True
        if fin:
            break
        partes.append(linea)
        j = j + 1
    return limpiar_texto("\n".join(partes))


def extraer_todo(carpeta_pdfs):
    resultado = []
    archivos = os.listdir(carpeta_pdfs)
    # los archivos empiezan por "1. ", "2. ", ...
    archivos.sort(key=lambda nombre: int(nombre.split(".")[0]))
    for nombre in archivos:
        texto = leer_texto_pdf(os.path.join(carpeta_pdfs, nombre))
        lector = PdfReader(os.path.join(carpeta_pdfs, nombre))
        meta = lector.metadata or {}
        resultado.append({
            "id": int(nombre.split(".")[0]),
            "archivo": nombre,
            "titulo_metadato": (meta.get("/Title") or "").strip(),
            "autor_metadato": (meta.get("/Author") or "").strip(),
            "abstract": extraer_abstract(texto),
        })
    return resultado


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datos = extraer_todo(os.path.join(base, "datos", "pdfs"))
    with open(os.path.join(base, "datos", "extraccion_automatica.json"), "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    for d in datos:
        print(d["id"], len(d["abstract"].split()), d["abstract"][:70], "...", d["abstract"][-50:])
