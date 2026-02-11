from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

messages = [
    SystemMessage(content="You are a helpful teaching assistant."),
    HumanMessage(content="Explain prompt engineering.")
]

response = llm.invoke(messages)
print(response.content)