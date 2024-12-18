import os
# แก้ไขปัญหา OpenMP และ symlinks warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def create_rag_pipeline(vectorstore):
    """สร้าง RAG pipeline"""
    # สร้าง LLM pipeline
    model_name = "bert-base-multilingual-cased"  # หรือโมเดลอื่นที่ต้องการใช้
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7
    )
    
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    # สร้าง QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
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
            print(f"- {doc.metadata.get('source', 'ไม่ระบุแหล่งท่มา')}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        print("คำตอบ: ไม่สามารถประมวลผลคำถามได้")