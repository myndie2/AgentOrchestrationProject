from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

prompt = "Explain what a Large Language Model is in one sentence."
response = llm.invoke(prompt)

print(response.content)