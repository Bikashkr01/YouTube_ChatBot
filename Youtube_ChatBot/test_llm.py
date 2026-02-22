from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", temperature=0.2)

response = llm.invoke("Explain Retrieval Augmented Generation in AI.")
print(response.content)