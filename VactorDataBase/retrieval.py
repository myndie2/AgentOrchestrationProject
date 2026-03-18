import numpy as np
from embedding import embed_query
# On importe la base que l'on a créée à l'étape précédente
from vector_store import vector_db

query = "I have eggs, flour, and honey. What can I make?"

# 1. On vectorise la requête (comme dans le guide)
query_vector = embed_query(query)

# 2. Simulation de db.similarity_search(query, k=1)
def similarity_search(q_vec, db, k=1):
    # On calcule le score de similarité pour chaque document
    # Plus le score est haut, plus c'est proche sémantiquement
    scored_docs = []
    for doc in db:
        score = np.dot(q_vec, doc["vector"])
        scored_docs.append((score, doc))
    
    # On trie par score décroissant et on prend les k meilleurs
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:k]]

# 3. Exécution de la recherche
docs = similarity_search(query_vector, vector_db, k=1)

# 4. Affichage du résultat (exactement comme ton guide)
print(f"Recette trouvée dans la base : {docs[0]['page_content']}")