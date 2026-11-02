# Pruebas de los algoritmos. Ejecutar desde la carpeta del proyecto:
#   python -m unittest discover pruebas -v

import os
import sys
import math
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from busqueda import kmp, boyer_moore, rabin_karp, ingenuo
from similitud import clasicos
from clustering import jerarquico


def todas_las_posiciones(texto, patron):
    # referencia sencilla con slicing (solo para probar)
    return [i for i in range(len(texto) - len(patron) + 1) if texto[i:i + len(patron)] == patron]


class PruebasBusqueda(unittest.TestCase):

    def test_tabla_prefijos(self):
        self.assertEqual(kmp.tabla_prefijos("ababaca"), [0, 0, 1, 2, 3, 0, 1])
        self.assertEqual(kmp.tabla_prefijos("aaaa"), [0, 1, 2, 3])

    def test_caracter_malo(self):
        t = boyer_moore.tabla_caracter_malo("example")
        self.assertEqual(t["e"], 6)
        self.assertEqual(t["x"], 1)

    def test_aleatorio_contra_referencia(self):
        random.seed(7)
        for _ in range(3000):
            alfabeto = random.choice(["ab", "abc", "acgt", "abcdefghij "])
            texto = "".join(random.choice(alfabeto) for _ in range(random.randint(0, 80)))
            patron = "".join(random.choice(alfabeto) for _ in range(random.randint(1, 6)))
            esperado = todas_las_posiciones(texto, patron)
            self.assertEqual(kmp.buscar(texto, patron)[0], esperado)
            self.assertEqual(boyer_moore.buscar(texto, patron)[0], esperado)
            self.assertEqual(rabin_karp.buscar(texto, patron)[0], esperado)
            # con un primo pequeno hay muchas colisiones, pero el resultado debe ser igual
            self.assertEqual(rabin_karp.buscar(texto, patron, q=13)[0], esperado)
            self.assertEqual(ingenuo.buscar(texto, patron)[0], esperado)

    def test_solapamientos(self):
        self.assertEqual(kmp.buscar("aaaaa", "aa")[0], [0, 1, 2, 3])
        self.assertEqual(boyer_moore.buscar("aaaaa", "aa")[0], [0, 1, 2, 3])
        self.assertEqual(rabin_karp.buscar("aaaaa", "aa")[0], [0, 1, 2, 3])


class PruebasSimilitud(unittest.TestCase):

    def test_levenshtein_conocidos(self):
        self.assertEqual(clasicos.levenshtein("kitten", "sitting")[0], 3)
        self.assertEqual(clasicos.levenshtein("", "abc")[0], 3)
        self.assertEqual(clasicos.levenshtein_distancia("flaw", "lawn"), 2)
        self.assertEqual(clasicos.levenshtein(["a", "b"], ["a", "c", "b"])[0], 1)

    def test_levenshtein_dos_versiones(self):
        random.seed(3)
        for _ in range(300):
            a = "".join(random.choice("abc") for _ in range(random.randint(0, 12)))
            b = "".join(random.choice("abc") for _ in range(random.randint(0, 12)))
            self.assertEqual(clasicos.levenshtein(a, b)[0], clasicos.levenshtein_distancia(a, b))

    def test_needleman_wunsch(self):
        puntaje, F, al_a, al_b = clasicos.needleman_wunsch("GATTACA", "GCATGCU")
        self.assertEqual(puntaje, 0)  # ejemplo clasico de Wikipedia
        self.assertEqual(len(al_a), len(al_b))
        # quitar los huecos devuelve las secuencias originales
        self.assertEqual("".join(x for x in al_a if x != "-"), "GATTACA")
        self.assertEqual("".join(x for x in al_b if x != "-"), "GCATGCU")

    def test_jaccard_y_coseno(self):
        self.assertAlmostEqual(clasicos.jaccard(["a", "b", "c"], ["b", "c", "d"]), 0.5)
        v1 = {"x": 1.0, "y": 0.0}
        v2 = {"x": 1.0, "y": 1.0}
        self.assertAlmostEqual(clasicos.coseno_dispersos(v1, v2), 1 / math.sqrt(2))

    def test_idf(self):
        idf, df = clasicos.calcular_idf([["a", "b"], ["a"], ["a", "c"]])
        self.assertEqual(idf["a"], 0.0)
        self.assertAlmostEqual(idf["b"], math.log(3))


class PruebasClustering(unittest.TestCase):

    def matriz_ejemplo(self):
        random.seed(11)
        puntos = [(random.random(), random.random()) for _ in range(12)]
        return [[math.dist(p, q) for q in puntos] for p in puntos]

    def test_contra_scipy(self):
        try:
            import numpy as np
            from scipy.cluster.hierarchy import linkage, cophenet
            from scipy.spatial.distance import squareform
        except ImportError:
            self.skipTest("scipy no instalado (solo se usa para verificar)")
        D = self.matriz_ejemplo()
        cond = squareform(np.array(D), checks=False)
        for metodo in ["single", "complete", "average"]:
            uniones = jerarquico.aglomerativo(D, metodo)
            Z = linkage(cond, method=metodo)
            self.assertTrue(np.allclose(sorted(u[2] for u in uniones), sorted(Z[:, 2])))
            self.assertAlmostEqual(jerarquico.correlacion_cofenetica(D, uniones), cophenet(Z, cond)[0], places=9)

    def test_silueta_contra_sklearn(self):
        try:
            import numpy as np
            from sklearn.metrics import silhouette_score
        except ImportError:
            self.skipTest("sklearn no instalado (solo se usa para verificar)")
        D = self.matriz_ejemplo()
        uniones = jerarquico.aglomerativo(D, "average")
        for k in [2, 3, 4]:
            etiquetas = jerarquico.cortar_arbol(uniones, len(D), k)
            ref = silhouette_score(np.array(D), etiquetas, metric="precomputed")
            self.assertAlmostEqual(jerarquico.silueta(D, etiquetas), ref, places=9)


if __name__ == "__main__":
    unittest.main()
