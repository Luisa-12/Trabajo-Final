# Análisis de algoritmos en el contexto de la bibliometría

Proyecto del curso Análisis de Algoritmos (2026-2) — Universidad del Quindío.
Aplicación web que procesa los abstracts de 20 artículos científicos sobre IA en educación y
implementa desde cero:

- **R1 – Similitud textual:** Levenshtein, Needleman-Wunsch, coseno TF-IDF y Jaccard (clásicos);
  promedio de vectores de palabras y Sentence-BERT (embeddings).
- **R2 – Clustering jerárquico:** single, complete y average linkage, con dendrogramas,
  correlación cofenética y coeficiente de silueta.
- **R3 – Búsqueda de patrones:** Knuth-Morris-Pratt, Boyer-Moore y Rabin-Karp, con ejecución
  paso a paso y medición de rendimiento.

## Estructura

```
app.py                     aplicación web (Flask)
corpus.py                  carga del corpus y cruce de metadatos
preprocesamiento.py        tokenización, stopwords y S-stemmer
extraccion/extractor_pdf.py  extracción del abstract desde los PDF
similitud/clasicos.py      Levenshtein, Needleman-Wunsch, TF-IDF + coseno, Jaccard
similitud/ia.py            similitud con embeddings (coseno y distancia euclidiana)
similitud/comparador.py    ejecuta los 6 algoritmos sobre los pares seleccionados
clustering/jerarquico.py   algoritmo aglomerativo, cofenética, silueta
clustering/dendrograma.py  dibujo del dendrograma
busqueda/kmp.py            Knuth-Morris-Pratt
busqueda/boyer_moore.py    Boyer-Moore (carácter malo + sufijo bueno)
busqueda/rabin_karp.py     Rabin-Karp (rolling hash)
busqueda/rendimiento.py    medición de tiempos y comparaciones
datos/                     PDF, corpus.json, metadatos y embeddings precalculados
scripts/                   generación de embeddings, casos de estudio y benchmark
pruebas/                   pruebas unitarias
web/                       plantillas HTML y estilos
```

## Ejecutar en local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                   # abrir http://127.0.0.1:5000
```

Pruebas:

```bash
python -m unittest discover pruebas -v
```

(Las pruebas de clustering se comparan contra `scipy` y `scikit-learn` si están instalados;
si no, se omiten. Esas librerías **no** se usan en la aplicación.)


## Regenerar los datos

```bash
python corpus.py                       # vuelve a extraer los abstracts de los PDF
pip install spacy sentence-transformers
python -m spacy download en_core_web_md
python scripts/generar_embeddings.py   # vectores de palabras y embeddings Sentence-BERT
python scripts/casos_estudio.py        # números de los casos de estudio del documento
python scripts/benchmark_busqueda.py   # experimentos de rendimiento del R3
```

Los embeddings incluidos en `datos/embeddings_minilm.json` se calcularon con la versión ONNX
cuantizada (8 bits) de `all-MiniLM-L6-v2`; al regenerarlos con `sentence-transformers` (modelo en
precisión completa) las similitudes pueden variar en el orden de ±0.01.
