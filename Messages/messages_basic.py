from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

response = llm.invoke([
    HumanMessage(content="What is an embedding?")
])

print(response.content)
