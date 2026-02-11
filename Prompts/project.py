from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

template = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question clearly:\n{question}"
)

response = llm.invoke(template.format(question="What is RAG?"))
print(response.content)