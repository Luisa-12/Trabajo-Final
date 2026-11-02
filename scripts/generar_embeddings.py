# Genera los datos de los dos modelos de IA del Requerimiento 1.
# Se ejecuta UNA vez (el corpus es estatico) y los resultados quedan en datos/.
# La aplicacion desplegada no necesita instalar estos modelos.
#
#   pip install spacy sentence-transformers
#   python -m spacy download en_core_web_md
#   python scripts/generar_embeddings.py

import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from corpus import cargar_corpus
from preprocesamiento import tokenizar, quitar_stopwords

corpus = cargar_corpus()

# ---------------- Modelo 1: vectores de palabras preentrenados (spaCy) ----------------
import spacy
nlp = spacy.load("en_core_web_md")
vocabulario = set()
for art in corpus:
    for palabra in quitar_stopwords(tokenizar(art["abstract"])):
        vocabulario.add(palabra)
vectores = {}
for palabra in sorted(vocabulario):
    lexema = nlp.vocab[palabra]
    if lexema.has_vector:
        vectores[palabra] = [round(float(x), 5) for x in lexema.vector]
print("Palabras del corpus con vector:", len(vectores), "de", len(vocabulario))
with open(os.path.join(BASE, "datos", "vectores_palabras.json"), "w") as f:
    json.dump({"modelo": "spaCy en_core_web_md - vectores de palabras estaticos (300 dimensiones)",
               "dimension": 300, "vectores": vectores}, f)

# ---------------- Modelo 2: Sentence-BERT ----------------
from sentence_transformers import SentenceTransformer
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
textos = [art["abstract"] for art in corpus]
embeddings = modelo.encode(textos, normalize_embeddings=True)
datos = {}
for art, e in zip(corpus, embeddings):
    datos[str(art["id"])] = [round(float(x), 6) for x in e]
with open(os.path.join(BASE, "datos", "embeddings_minilm.json"), "w") as f:
    json.dump({"modelo": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384,
               "max_tokens": modelo.max_seq_length, "pooling": "promedio (mean pooling) + normalizacion L2",
               "embeddings": datos}, f)
print("Embeddings guardados para", len(datos), "abstracts")
