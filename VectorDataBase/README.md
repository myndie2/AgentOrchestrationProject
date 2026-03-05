#  Vector Database

## C'est quoi une Vector Database ?

Une base de données vectorielle stocke des textes sous forme de **vecteurs numériques** (embeddings). Contrairement à une base SQL qui cherche des mots exacts, elle permet des **recherches sémantiques** : trouver des documents qui ont le même *sens*, pas forcément les mêmes mots.

**Caractéristiques communes :**
- Semantic search — chercher par sens, pas par mot-clé exact
- RAG (Retrieval-Augmented Generation) — enrichir un LLM avec tes propres données
- Store embeddings + metadata — chaque document = vecteur + texte original
- Chunked documents — les textes sont découpés en morceaux avant d'être vectorisés

---

## La chaîne RAG

```
[Tes recettes]
     │
     ▼
[embedding.py] ──── transforme chaque texte en vecteur numérique
     │
     ▼
[vector_store.py] ── stocke vecteurs + métadonnées dans une base locale
     │
     ▼
[retrieval.py] ───── cherche les documents les plus proches de ta question
     │
     ▼
[rag_pipeline.py] ── donne le contexte trouvé au LLM pour qu'il réponde
     │
     ▼
[Réponse du Chef 👨‍🍳]
```

**Sans RAG** → le LLM répond de sa mémoire générale  
**Avec RAG** → le LLM s'appuie sur *tes* données spécifiques

---

## Structure des fichiers

```
VectorDataBase/
├── embedding.py       # Étape 1 : générer des vecteurs
├── vector_store.py    # Étape 2 : stocker les vecteurs
├── retrieval.py       # Étape 3 : chercher les documents pertinents
└── rag_pipeline.py    # Étape 4 : pipeline RAG complet
```

---

## Choix techniques

| Rôle | Cours (LangChain/OpenAI) | Notre version (gratuite) |
|---|---|---|
| Embeddings | `OpenAIEmbeddings()` | API HuggingFace Router |
| Vector Store | `FAISS.from_texts()` | Implémentation manuelle numpy |
| Similarity Search | `db.similarity_search()` | `np.dot()` manuel |
| LLM | `ChatOpenAI / gpt-4` | `meta-llama/Meta-Llama-3-8B-Instruct` |

---

## Étape 1 — `embedding.py`

### À quoi ça sert ?
Transformer un texte en vecteur numérique. C'est la fondation de tout le système : sans bons vecteurs, la recherche sémantique ne fonctionne pas.

### Équivalent du cours
```python
# Cours (LangChain)
embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("What is a vector database?")
```

### Notre version
```python
# embedding.py
import os, requests, numpy as np
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def embed_query(text: str) -> list:
    """
    Génère un vrai vecteur sémantique via HuggingFace Router.
    Équivalent de OpenAIEmbeddings().embed_query() du cours.
    """
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )
    return response.json()
```

### Pourquoi ce modèle ?
`sentence-transformers/all-MiniLM-L6-v2` est un modèle spécialisé dans la génération d'embeddings (pas un modèle de chat). Il produit des vecteurs de dimension defini où la proximité géométrique reflète la proximité sémantique.

### Preuve que ça fonctionne
```
Taille du vecteur     : 384
Similarité cuisine/cuisine : 0.471   ← élevé
Similarité cuisine/voiture : 0.125   ← bas
```

---

## Étape 2 — `vector_store.py`

### À quoi ça sert ?
Indexer une collection de documents en les transformant tous en vecteurs et en les stockant avec leur texte original (métadonnée).

### Équivalent du cours
```python
# Cours (LangChain + FAISS)
db = FAISS.from_texts(texts, embeddings)
```

### Notre version
```python
# vector_store.py
from embedding import embed_query

texts = [
    "Pancakes: flour, eggs, milk, sugar. Mix and fry.",
    "Honey Cookies: flour, honey, eggs. Bake at 180°C.",
    "Omelette: eggs, milk, salt. Fry in a pan."
]

vector_db = []  # notre "FAISS" maison

for t in texts:
    vector = embed_query(t)
    vector_db.append({
        "page_content": t,   # métadonnée = texte original
        "vector": vector     # embedding = vecteur numérique
    })
```

### Ce qu'on stocke pour chaque document
```
{
  "page_content": "Pancakes: flour, eggs, milk, sugar. Mix and fry.",
  "vector": [0.023, -0.031, -0.117, ...]   ← 384 nombres
}
```

### Résultat attendu
```
Base de données culinaire prête : 3 recettes indexées.
- Stocké: Pancakes...       (Vecteur de taille 384) 
- Stocké: Honey Cookies...  (Vecteur de taille 384) 
- Stocké: Omelette...       (Vecteur de taille 384) 
```

---

## Étape 3 — `retrieval.py`

### À quoi ça sert ?
Trouver les documents les plus proches sémantiquement d'une requête. C'est le moteur de recherche du RAG.

### Équivalent du cours
```python
# Cours (LangChain)
docs = db.similarity_search(query, k=1)
print(docs[0].page_content)
```

### Notre version
```python
# retrieval.py
import numpy as np
from embedding import embed_query
from vector_store import vector_db

def similarity_search(q_vec, db, k=1):
    """
    Équivalent de db.similarity_search() du cours.
    Calcule la similarité cosinus entre la requête et chaque document.
    """
    scored_docs = []
    for doc in db:
        score = np.dot(q_vec, doc["vector"])   # produit scalaire = similarité
        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)  # trier par score
    return [doc for score, doc in scored_docs[:k]]       # garder les k meilleurs
```

### Comment fonctionne `np.dot()` ?
Le produit scalaire entre deux vecteurs mesure leur similarité directionnelle :
- Score proche de **1.0** → documents très similaires
- Score proche de **0.0** → documents très différents

### Résultat attendu
```
Requête : "I have eggs, flour, and honey. What can I make?"
Trouve  : "Honey Cookies: flour, honey, eggs. Bake at 180°C."
```
La recherche trouve Honey Cookies et pas Omelette, car les ingrédients (flour, honey, eggs) correspondent mieux.

---

## Étape 4 — `rag_pipeline.py`

### À quoi ça sert ?
Assembler toute la chaîne : vectoriser la question → chercher les recettes pertinentes → donner le contexte au LLM → obtenir une réponse ancrée dans nos données.

### Équivalent du cours
```python
# Cours (LangChain)
docs = db.similarity_search("Explain vector databases", k=2)
context = "\n".join([d.page_content for d in docs])
response = llm.invoke(f"Use the context below to answer:\n{context}")
```

### Notre version
```python
# rag_pipeline.py
from openai import OpenAI
from retrieval import similarity_search, vector_db
from embedding import embed_query

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

# 1. Vectoriser la question
query = "I have eggs, flour, and honey. What can I make?"
query_vec = embed_query(query)

# 2. Chercher les recettes pertinentes (Retrieval)
docs = similarity_search(query_vec, vector_db, k=2)
context = "\n".join([d["page_content"] for d in docs])

# 3. Appeler le LLM avec le contexte (Augmented Generation)
messages = [
    {"role": "system", "content": "You are a head chef. Use ONLY the context provided."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
]

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=messages
)
print(response.choices[0].message.content)
```

### Les 3 étapes du RAG en une image
```
Question utilisateur
        │
        ▼
  [embed_query()]          ← transforme la question en vecteur
        │
        ▼
  [similarity_search()]    ← trouve les 2 recettes les plus proches
        │
        ▼
  [LLM + contexte]         ← le chef répond en s'appuyant sur ces recettes
        │
        ▼
  Réponse ancrée dans tes données
```

---

## Comment tester chaque étape

```bash
# Étape 1 : vérifier que les embeddings fonctionnent
python embedding.py

# Étape 2 : vérifier que la base se construit
python vector_store.py
# Attendu : 3 recettes indexées avec vecteurs de taille du vecteur embedding

# Étape 3 : vérifier que la recherche est sémantique
python retrieval.py
# Attendu : Honey Cookies trouvé pour une question sur eggs/flour/honey

# Étape 4 : tester la chaîne complète
python rag_pipeline.py
# Attendu : réponse du chef basée sur les recettes de la base
```

---

## Dépendances

```
requests    # appels API HuggingFace
numpy       # calcul de similarité (np.dot)
openai      # client LLM via HuggingFace Router
python-dotenv  # gestion du token HF_TOKEN
sentence-transformers #génère vecteur en local et gratuit a la place de openaiEmbedding
```


Installation :
```bash
pip install requests numpy openai python-dotenv sentence-transformers
```

---

## Sources

- [HuggingFace Router Documentation](https://router.huggingface.co)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [RAG — Retrieval Augmented Generation](https://huggingface.co/blog/rag)
