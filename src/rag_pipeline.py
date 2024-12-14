import os
# แก้ไขปัญหา OpenMP และ symlinks warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import torch

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    print("กำลังโหลดโมเดลและ tokenizer...")
    model_name = "google/flan-t5-small"
    
    # สร้าง pipeline โดยตรงจาก transformers พร้อมตั้งค่าที่เหมาะสม
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        max_length=512,
        do_sample=True,  # เปิดใช้งาน sampling
        temperature=0.7,
        top_p=0.9,
        device="cpu",
        model_kwargs={
            "torch_dtype": torch.float32
        }
    )
    
    # แปลง pipeline เป็น LangChain LLM
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.7}
    )
    
    print("กำลังสร้าง QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    print("สร้าง RAG pipeline เสร็จสมบูรณ์")
    return qa_chain

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("กำลังโหลด vectorstore...")
    vectorstore = FAISS.load_local(
        "vectorstore", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    qa_chain = create_rag_pipeline(vectorstore)
    
    query = "What is data quality?"
    print(f"\nคำถามทดสอบ: {query}")
    try:
        result = qa_chain.invoke(query)
        print(f"\nคำตอบ: {result['result']}")
        print("\nแหล่งข้อมูล:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata.get('source', 'ไม่ระบุแหล่งท��่มา')}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        print("คำตอบ: ไม่สามารถประมวลผลคำถามได้")