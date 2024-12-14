from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# แก้ไขปัญหา OpenMP
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def create_vectorstore(pdf_path: str, save_path: str = "vectorstore"):
    # ตรวจสอบว่าไฟล์มีอยู่จริง
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ที่: {pdf_path}")
        
    # โหลดเอกสาร PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # แบ่งเอกสารเป็นชิ้นเล็กๆ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    try:
        # สร้าง embeddings โดยใช้ HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # สร้าง vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # บันทึก vectorstore
        vectorstore.save_local(save_path)
        print(f"บันทึก vectorstore ไปที่: {save_path}")
        
        return vectorstore
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        raise

if __name__ == "__main__":
    pdf_path = r"data/raw_pdfs/Data+Quality+Masterclass+Guide.pdf"
    vectorstore = create_vectorstore(pdf_path)