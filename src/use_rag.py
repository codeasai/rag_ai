import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import create_rag_pipeline

# แก้ไขปัญหา OpenMP (ถ้ามี)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def load_vectorstore(vectorstore_path: str = "vectorstore"):
    # สร้าง embeddings โดยใช้ HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # โหลด vectorstore พร้อมยืนยันความปลอดภัย
    vectorstore = FAISS.load_local(
        vectorstore_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def query_documents(qa_chain, query: str) -> str:
    try:
        print(f"\nคำถาม: {query}")
        print("กำลังค้นหาคำตอบ...")
        
        # ใช้ invoke แทน run
        response = qa_chain.invoke(query)
        
        print("\nคำตอบ:", response['result'])
        print("\nแหล่งข้อมูล:")
        for doc in response['source_documents']:
            print(f"- {doc.metadata.get('source', 'ไม่ระบุแหล่งที่มา')}")
        
        return response['result']
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        return "ไม่สามารถประมวลผลคำถามได้"

if __name__ == "__main__":
    # โหลด vectorstore
    vectorstore = load_vectorstore()
    
    # สร้าง qa_chain
    qa_chain = create_rag_pipeline(vectorstore)
    
    # ตัวอย่างการใช้งาน
    query = "คำถามที่ต้องการถาม?"
    answer = query_documents(qa_chain, query)
    print(f"คำตอบ: {answer}") 