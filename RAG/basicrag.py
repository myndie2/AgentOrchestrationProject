import os
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# ==========================================
# PHASE D'INGESTION (Préparation des données)
# ==========================================

# 1. Load a pdf document
def load_pdf(file_path):
    # Remplacer par le chemin de votre livre de recettes PDF
    # Pour l'exemple, on simule le chargement du document
    return "Recette 1 : Riz aux haricots et lait. Cuire le riz. Chauffer le lait avec les haricots. Mélanger."

documents = load_pdf("livre_recettes.pdf")

# 2. Split the document into smaller chunks
def recursive_character_text_splitter(text, chunk_size=500, chunk_overlap=100):
    """Découpage simple s'inspirant du RecursiveCharacterTextSplitter"""
    docs = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        docs.append(text[start:end])
        start += chunk_size - chunk_overlap
    return docs

docs = recursive_character_text_splitter(documents, chunk_size=500, chunk_overlap=100)

# 3. Create embeddings for the document chunks
# Équivalent de HuggingFaceEmbeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_list):
    return embedding_model.encode(text_list).tolist()

# 4. Store the embeddings in a vector store (chroma in this case)
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(name="rag_collection")

# Stockage effectif
if docs:
    vector_store.add(
        documents=docs,
        embeddings=get_embeddings(docs),
        ids=[f"doc_{i}" for i in range(len(docs))]
    )


# ==========================================
# PHASE DE GÉNÉRATION (Le pipeline RAG)
# ==========================================

# 5. Create a retriever from the vector store
def retriever(query, k=2):
    """Recherche dans ChromaDB (similaire au vector_store.as_retriever)"""
    query_embedding = get_embeddings([query])
    results = vector_store.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    # Retourne la liste des documents trouvés
    return results['documents'][0] if results['documents'] else []

# 6. Initialize LLM (Notre client OpenAI vers Hugging Face)
client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# 7. Create prompt template
def prompt_template(context, question):
    """Équivalent de la classe PromptTemplate"""
    return f"""You are an AI chef assistant. Use the following context to suggest a recipe.
If the answer is not in the context, say you don't know.
Context: {context}
Question: What can I cook with {question}?
"""

# 8. OutputParser definition
def str_output_parser(response):
    """Équivalent du StrOutputParser() : extrait juste le texte de la réponse API"""
    return response.choices[0].message.content.strip()


# 9. Create RAG chain
def format_docs(retrieved_docs):
    """Assemble les chunks récupérés en un seul texte"""
    return "\n\n".join(retrieved_docs)

def rag_chain_invoke(query):
    """
    Voici la magie ! L'équivalent exact de :
    retriever | format_docs | prompt | llm | parser
    """
    # a. retriever
    retrieved_docs = retriever(query, k=2)
    
    # b. format_docs
    context = format_docs(retrieved_docs)
    
    # c. prompt
    final_prompt = prompt_template(context, query)
    
    # d. llm (Appel au modèle)
    messages = [
        {"role": "system", "content": "You are a helpful culinary assistant."},
        {"role": "user", "content": final_prompt}
    ]
    llm_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7
    )
    
    # e. parser
    final_output = str_output_parser(llm_response)
    
    return final_output


# ==========================================
# EXÉCUTION
# ==========================================

# 10. Ask a question
if __name__ == "__main__":
    query = "milk and beans"
    print(f"Question : Que puis-je cuisiner avec {query} ?\n")
    
    response = rag_chain_invoke(query)
    
    print("--- RÉPONSE DU RAG ---")
    print(response)