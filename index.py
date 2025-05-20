from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Texto base
texto_base = """
A inteligência artificial (IA) é um campo da ciência da computação dedicado à criação de sistemas que podem realizar tarefas normalmente associadas à inteligência humana. Alguns exemplos incluem reconhecimento de fala, visão computacional, tomada de decisões e tradução de idiomas.
A IA moderna depende fortemente de algoritmos de aprendizado de máquina, onde os computadores aprendem com grandes volumes de dados.
"""

# 2. Dividir texto em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.create_documents([texto_base])

# 3. Criar embeddings usando modelo HuggingFace local
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Criar o banco vetorial
vectorstore = FAISS.from_documents(docs, embedding_model)

# 5. Criar o modelo LLM local com Ollama
llm = OllamaLLM(model="llama3.1:8b") # <- Mudar conforme LLM instalada no seu ambiente 

# 6. Pipeline de Perguntas e Respostas
qa = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=vectorstore.as_retriever(),
  return_source_documents=True
)

question = "Quais são os exemplos de aplicações de inteligência artificial?"
# question = "Quem é o principal criador da Linguagem Artificial" # Pergunta que não tem haver com a fonte
response = qa.invoke(question)


print("\n📌 Pergunta:", question)
print("\n✅ Resposta:", response['result'])
print("\n📚 Fontes:")
for doc in response["source_documents"]:
  print("-", doc.page_content)
