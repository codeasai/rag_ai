import os
import logging
from pathlib import Path
import yaml
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_pipeline import create_rag_pipeline

# แก้ไขปัญหา OpenMP
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def load_config(config_path: str = "config.yaml") -> dict:
    """โหลด config file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"ไม่สามารถโหลด config file: {str(e)}")
        return {}

def setup_logging(log_dir: str = "logs"):
    """ตั้งค่า logging"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/query_rag.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_vectorstore(
    model_type: str = "pretrained",
    vectorstore_path: str = "vectorstore",
    custom_model_path: Optional[str] = None
) -> FAISS:
    """
    โหลด vectorstore
    
    Args:
        model_type: "pretrained" หรือ "custom"
        vectorstore_path: path ของ vectorstore
        custom_model_path: path ของโมเดลที่เทรนเอง (ถ้าใช้ custom)
    """
    try:
        # เลือกโมเดลตามประเภท
        if model_type == "pretrained":
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logging.info("ใช้โมเดลสำเร็จรูป: sentence-transformers/all-MiniLM-L6-v2")
        else:
            if not custom_model_path:
                custom_model_path = "models/model.safetensors"
            embeddings = HuggingFaceEmbeddings(
                model_name=custom_model_path
            )
            logging.info(f"ใช้โมเดลที่เทรนเอง: {custom_model_path}")
        
        # โหลด vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logging.info(f"โหลด vectorstore จาก: {vectorstore_path}")
        return vectorstore
        
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการโหลด vectorstore: {str(e)}")
        raise

def query_documents(qa_chain, query: str) -> str:
    """ถามคำถามกับเอกสาร"""
    try:
        print(f"\nคำถาม: {query}")
        print("กำลังค้นหาคำตอบ...")
        
        response = qa_chain.invoke(query)
        
        print("\nคำตอบ:", response['result'])
        print("\nแหล่งข้อมูล:")
        for doc in response['source_documents']:
            print(f"- {doc.metadata.get('source', 'ไม่ระบุแหล่งที่มา')}")
        
        return response['result']
        
    except Exception as e:
        error_msg = f"เกิดข้อผิดพลาด: {str(e)}"
        logging.error(error_msg)
        return error_msg

def main():
    # โหลด config และตั้งค่า logging
    config = load_config()
    setup_logging(config.get('log_dir', 'logs'))
    
    try:
        # ให้ผู้ใช้เลือกประเภทโมเดล
        print("\nเลือกประเภทโมเดลที่ต้องการใช้:")
        print("1. โมเดลสำเร็จรูป (sentence-transformers/all-MiniLM-L6-v2)")
        print("2. โมเดลที่เทรนเอง")
        
        choice = input("เลือกหมายเลข (1 หรือ 2): ").strip()
        
        # โหลด vectorstore ตามที่เลือก
        if choice == "1":
            vectorstore = load_vectorstore(model_type="pretrained")
        else:
            custom_path = input("\nระบุ path ของโมเดล (กด Enter เพื่อใช้ค่าเริ่มต้น models/model.safetensors): ").strip()
            vectorstore = load_vectorstore(
                model_type="custom",
                custom_model_path=custom_path if custom_path else None
            )
        
        # สร้าง qa_chain
        qa_chain = create_rag_pipeline(vectorstore)
        
        # เริ่มการถาม-ตอบ
        print("\nพร้อมรับคำถาม (พิมพ์ 'exit' เพื่อออก)")
        while True:
            query = input("\nคำถาม: ").strip()
            if query.lower() == 'exit':
                break
            
            answer = query_documents(qa_chain, query)
            logging.info(f"คำถาม: {query}")
            logging.info(f"คำตอบ: {answer}")
            
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาด: {str(e)}")
        raise

if __name__ == "__main__":
    main() 