from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def prepare_documents_for_rag(processed_documents):
    # แบ่งเอกสารเป็นชิ้นเล็กๆ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(processed_documents)
    
    # สร้าง embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="models/model.safetensors"
    )
    
    # สร้าง vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore 