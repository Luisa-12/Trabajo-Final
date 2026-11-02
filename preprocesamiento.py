# Preprocesamiento de los abstracts.
# Pasos: minusculas -> tokenizacion -> quitar stopwords -> raiz simple (S-stemmer)
# Todo se hace recorriendo los caracteres con ciclos, sin librerias de NLP.

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "although", "am", "among", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "e",
    "eg", "et", "al", "etc", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in",
    "into", "is", "it", "its", "itself", "just", "may", "me", "might", "more", "most", "much", "must",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "thus", "to", "too", "under", "until", "up", "upon", "us", "very", "was", "we",
    "were", "what", "when", "where", "whereof", "whether", "which", "while", "who", "whom", "why",
    "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
    "s", "t", "ii", "iii", "iv", "vi", "well", "via", "many", "since", "like", "able", "rather",
}


def es_letra_o_digito(c):
    return ("a" <= c <= "z") or ("0" <= c <= "9") or c in "áéíóúñüçäöšć"


def tokenizar(texto):
    """Divide el texto en palabras. Un token es una secuencia de letras o digitos."""
    texto = texto.lower()
    tokens = []
    actual = ""
    for c in texto:
        if es_letra_o_digito(c):
            actual = actual + c
        else:
            if actual != "":
                tokens.append(actual)
            actual = ""
    if actual != "":
        tokens.append(actual)
    return tokens


def quitar_stopwords(tokens):
    resultado = []
    for t in tokens:
        # se quitan stopwords, numeros sueltos y tokens de una letra
        if t not in STOPWORDS and len(t) > 1 and not t.isdigit():
            resultado.append(t)
    return resultado


def raiz_simple(palabra):
    """S-stemmer (Harman, 1991): solo normaliza plurales en ingles.
    studies -> study, analyses -> analyse, systems -> system"""
    if len(palabra) <= 3:
        return palabra
    if palabra.endswith("ies") and not palabra.endswith("eies") and not palabra.endswith("aies"):
        return palabra[:-3] + "y"
    if palabra.endswith("es") and not palabra.endswith("aes") and not palabra.endswith("ees") and not palabra.endswith("oes"):
        return palabra[:-1]
    if palabra.endswith("s") and not palabra.endswith("us") and not palabra.endswith("ss"):
        return palabra[:-1]
    return palabra


def preprocesar(texto, usar_raiz=True):
    """Devuelve la lista de terminos limpios del texto."""
    tokens = quitar_stopwords(tokenizar(texto))
    if usar_raiz:
        nuevos = []
        for t in tokens:
            nuevos.append(raiz_simple(t))
        tokens = nuevos
    return tokens
