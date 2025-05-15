from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAI

document_1 = Document(page_content="Quedas abruptas de penhascos, ocorridas durante perseguições a alvos que se movimentam a velocidades acima da média da espécie.", metadata={"baz": "bar"})
document_2 = Document(page_content="Explosões causadas por produtos ACME, desde que utilizados conforme especificações do catálogo oficial e acompanhadas de nota fiscal válida.", metadata={"bar": "baz"})
document_3 = Document(page_content="Colisões frontais com paredes pintadas como túneis, desde que o túnel não exista fisicamente no local.")
documents = [document_1, document_2, document_3]

embedding = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embedding)

qa = RetrievalQA.from_chain_type(llm=OpenAI(model='gpt-4.1-nano'), retriever=vector_store.as_retriever())
result = qa.invoke("O seguro cobre quedas de penhascos?")

print(f"Pergunta: {result['query']}")
print(f"Resposta: {result['result']}")