from fastapi import FastAPI, File, UploadFile
import os
from dotenv import load_dotenv
import tempfile
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
#from langchain_docling.loader import DoclingLoader
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

load_dotenv()
app = FastAPI()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "startnew"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
model = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate(
    template="You are a helpful assistant that helps people find information. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. This is the Context i retrived from RAG : {context} ,  Question: {question}",
    input_variables=["context", "question"]
)
parser = StrOutputParser()
vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chain = prompt | model | parser

@app.get("/health")
async def health_check():
    return {"status": "healthy"}



@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    #storing the file in a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(
        temp_file_path,
        mode="page",
        images_inner_format="markdown-img",
        images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o-mini")),
    )

    documents = loader.load()
    os.remove(temp_file_path)
    # add the documents to the vector store
    documents = text_splitter.split_documents(documents)
    vector_store.add_documents(documents)
    return {"status": "success", "message": f"Uploaded {len(documents)} documents."}


chat_history = []
chunks = []

@app.post("/ask/")
async def ask_question(question: str):
    global chat_history
    global chunks
    chat_history.append({"role": "user", "content": question})
    retriever = vector_store.as_retriever()
    compressor = RankLLMRerank(top_n=5, model="gpt", gpt_model="gpt-4o-mini")
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
    compressed_docs = compression_retriever.invoke(question)
    chunks.append(compressed_docs)
    context = "\n\n".join([doc.page_content for doc in compressed_docs])
    answer = chain.invoke({"context": context, "question": question})
    chat_history.append({"role": "assistant", "content": answer})
    return {"answer": answer}



@app.get("/chat_history/")
async def get_chat_history():
    return {"chat_history": chat_history}

@app.get("/chunks/")
async def get_chunks():
    return {"chunks": chunks}