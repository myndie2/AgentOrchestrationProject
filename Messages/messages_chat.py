from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

messages = [
    SystemMessage(content="You explain concepts briefly."),
    HumanMessage(content="What is a vector database?")
]

response = llm.invoke(messages)
print(response.content)