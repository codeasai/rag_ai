import os
from pathlib import Path
import logging
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader

# แก้ไขปัญหา OpenMP
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def setup_logging():
    """ตั้งค่า logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_documents(pdf_dir: str = "data/raw_pdfs") -> List:
    """โหลดเอกสาร PDF"""
    documents = []
    pdf_dir = Path(pdf_dir)
    
    logging.info(f"กำลังโหลดเอกสารจาก {pdf_dir}")
    
    # โหลดทุกไฟล์ PDF ในโฟลเดอร์
    for pdf_path in pdf_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
            logging.info(f"โหลด {pdf_path.name} สำเร็จ")
        except Exception as e:
            logging.error(f"ไม่สามารถโหลด {pdf_path.name}: {str(e)}")
    return documents

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """แบ่งเอกสารเป็นชิ้นเล็กๆ"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    logging.info(f"แบ่งเอกสารเป็น {len(texts)} ส่วน")
    return texts

def create_vectorstore(
    texts: List,
    model_type: str = "pretrained",
    custom_model_path: Optional[str] = None,
    save_path: str = "vectorstore"
):
    """สร้าง vectorstore"""
    try:
        # เลือกโมเดลตามประเภท
        if model_type == "pretrained":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logging.info(f"ใช้โมเดลสำเร็จรูป: {model_name}")
        else:
            model_name = custom_model_path or "models/model.safetensors"
            logging.info(f"ใช้โมเดลที่เทรนเอง: {model_name}")
            
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # สร้างและบันทึก vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(save_path)
        logging.info(f"บันทึก vectorstore ไปที่ {save_path}")
        
        return vectorstore
        
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการสร้าง vectorstore: {str(e)}")
        raise

def main():
    setup_logging()
    
    try:
        # ให้ผู้ใช้เลือกประเภทโมเดล
        print("\nเลือกประเภทโมเดลที่ต้องการใช้:")
        print("1. โมเดลสำเร็จรูป (sentence-transformers/all-MiniLM-L6-v2)")
        print("2. โมเดลที่เทรนเอง")
        
        choice = input("เลือกหมายเลข (1 หรือ 2): ").strip()
        
        # โหลดและแบ่งเอกสาร
        documents = load_documents()
        texts = split_documents(documents)
        
        # สร้าง vectorstore
        if choice == "1":
            create_vectorstore(texts, model_type="pretrained")
        else:
            custom_path = input("\nระบุ path ของโมเดล (กด Enter เพื่อใช้ค่าเริ่มต้น models/model.safetensors): ").strip()
            create_vectorstore(
                texts,
                model_type="custom",
                custom_model_path=custom_path if custom_path else None
            )
            
        print("\nสร้าง vectorstore เสร็จสมบูรณ์")
        
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาด: {str(e)}")
        raise

if __name__ == "__main__":
    main()