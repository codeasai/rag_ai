import logging
from pathlib import Path
import fitz  # PyMuPDF
import json
import re
from typing import List, Dict
from tqdm import tqdm

from utils import load_config, setup_logging


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, str]]:
    """
    ดึงข้อความจาก PDF และแบ่งเป็นย่อหน้า
    """
    documents = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # แบ่งเป็นย่อหน้า
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) > 50:  # เก็บเฉพาะย่อหน้าที่มีความยาวพอสมควร
                    documents.append({
                        'source': pdf_path.name,
                        'page': page_num + 1,
                        'text': para
                    })
        
        return documents
    
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {pdf_path}: {str(e)}")
        return []


def clean_text(text: str) -> str:
    """
    ทำความสะอาดข้อความ
    """
    # ลบช่องว่างที่ไม่จำเป็น
    text = re.sub(r'\s+', ' ', text)
    # ลบตัวอักษรพิเศษ
    text = re.sub(r'[^\w\s\u0E00-\u0E7F]', ' ', text)
    return text.strip()


def process_pdfs(config: dict):
    """
    ประมวลผลไฟล์ PDF ทั้งหมดและบันทึกผล
    """
    pdf_dir = Path(config['data']['pdf_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_documents = []
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    if not pdf_files:
        logging.error(f"ไม่พบไฟล์ PDF ใน {pdf_dir}")
        return

    logging.info(f"เริ่มประมวลผล PDF จำนวน {len(pdf_files)} ไฟล์")
    
    for pdf_path in tqdm(pdf_files, desc="กำลังประมวลผล PDF"):
        documents = extract_text_from_pdf(pdf_path)
        
        # ทำความสะอาดข้อ���วาม
        for doc in documents:
            doc['text'] = clean_text(doc['text'])
        
        all_documents.extend(documents)

    # บันทึกผลลัพธ์
    output_path = processed_dir / 'processed_documents.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)

    logging.info(f"ประมวลผลเสร็จสิ้น: {len(all_documents)} ย่อหน้า")
    logging.info(f"บันทึกผลไปที่: {output_path}")


def main():
    config = load_config('config.yaml')
    setup_logging(config['log_dir'])
    
    try:
        process_pdfs(config)
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาด: {str(e)}")
        raise


if __name__ == '__main__':
    main()
