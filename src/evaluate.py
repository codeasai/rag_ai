import logging
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from utils import load_config, setup_logging

def load_test_data(processed_dir: Path):
    """โหลดข้อมูลทดสอบ"""
    data_path = processed_dir / 'processed_documents.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # ใช้ข้อมูล 20% สุดท้ายเป็นชุดทดสอบ
    test_size = int(len(documents) * 0.2)
    test_documents = documents[-test_size:]
    
    texts = [doc['text'] for doc in test_documents]
    # สมมติว่าเป็นงาน binary classification เหมือนในการเทรน
    labels = [1 if len(doc['text']) > 200 else 0 for doc in test_documents]
    
    return texts, labels

def evaluate_model(config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"ใช้งานอุปกรณ์: {device}")

    # โหลดโมเดลและ tokenizer
    model_path = Path(config['model_dir']) / 'trained_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    # โหลดข้อมูลทดสอบ
    processed_dir = Path(config['data']['processed_dir'])
    test_texts, test_labels = load_test_data(processed_dir)
    
    # ทำนาย
    model.eval()
    predictions = []
    
    for text in tqdm(test_texts, desc="กำลังประเมินผล"):
        inputs = tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1)
            predictions.append(pred.item())

    # คำนวณและแสดงผลการประเมิน
    logging.info("\nผลการประเมิน:")
    logging.info("\nClassification Report:")
    print(classification_report(test_labels, predictions))
    
    logging.info("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predictions))

def main():
    config = load_config('config.yaml')
    setup_logging(config['log_dir'])
    
    try:
        evaluate_model(config)
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาด: {str(e)}")
        raise

if __name__ == '__main__':
    main() 