from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

conversation = [
    HumanMessage(content="What is RAG?"),
    AIMessage(content="RAG combines retrieval with generation."),
    HumanMessage(content="Why is it useful?")
]

response = llm.invoke(conversation)
print(response.content)