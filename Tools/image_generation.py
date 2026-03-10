# Permet à un modèle de langage de déclencher une génération d’image à partir d’une demande utilisateur en utilisant Stable Diffusion XL.

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
from PIL import Image

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

# Génère une image via Stable Diffusion XL
def generate_image(prompt: str) -> str:
    img_client = InferenceClient(token=os.getenv("HF_TOKEN")) # on crée un client Hugging Face spécialisé pour l’inférence
    image = img_client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0"
    )
    image.save("output.png")
    return "Image sauvegardée dans 'output.png'"



tools = [{
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Génère une image à partir d'une description textuelle.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Description détaillée en anglais."}
            },
            "required": ["prompt"]
        }
    }
}]



def ask_AI(question):
    messages = [
        {"role": "system", "content": "Tu es un assistant créatif. Quand l'utilisateur demande une image, appelle generate_image avec un prompt détaillé en anglais."},
        {"role": "user", "content": question}
    ]

    # ÉTAPE A : Premier appel (L'IA analyse la question
    # Cas 1: le modèle répond directement
    # Cas 2: le modèle demande un appel d'outil
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        tools=tools
    )

    # récupération du message du modèle (soit le premier message généré)
    msg = response.choices[0].message

    # ÉTAPE B : Exécution de l'outil
    if msg.tool_calls: # le modèle ne veut pas encore répondre directement ; il veut d’abord utiliser un outil
        messages.append(msg) # On ajoute la décision du modèle à l’historique, important car pour le second appel, le modèle doit “se souvenir” qu’il a demandé un outil
        for tool_call in msg.tool_calls:
            # On décode la recherche suggérée par l'IA
            args = json.loads(tool_call.function.arguments) # on renvoie les arguments sous forme de chaine json
            print(f" Génération d'image : {args['prompt']}")

            # On lance la recherche réelle
            result = generate_image(args["prompt"])

            # On injecte les résultats du web dans la conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id, # id de l'appel
                "content": result
            })

        # ÉTAPE C : Réponse finale
        final_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages
        )
        return final_response.choices[0].message.content

    return msg.content




if __name__ == "__main__":
    print("\n--- RÉPONSE ---\n")
    print(ask_AI("Génère une image d'un plat de pâtes élégant et coloré, tendance 2026."))



# Étape 0
# Le programme définit un outil de génération d’image.
# La fonction generate_image() utilise Stable Diffusion XL via HuggingFace.
# Elle génère une image à partir d’un prompt texte.
# L’image est sauvegardée dans le fichier output.png.

# Étape 1
# L’utilisateur pose une question.

# Étape 2
# Le modèle reçoit :
# la question
# la consigne système
# la liste des outils disponibles (generate_image)


# Étape 3
# Le modèle choisit :
# soit répondre avec du texte
# soit appeler l’outil generate_image

# Étape 4
# Le code Python exécute vraiment l’outil.
# La fonction :
# generate_image(prompt)
# appelle Stable Diffusion XL
# génère une image
# sauvegarde l’image dans output.png

# Étape 5
# Le code renvoie le résultat de l’outil au modèle dans l’historique.

# Étape 6
# Le modèle lit ce résultat et rédige la réponse finale.