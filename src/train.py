import logging
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import load_config, setup_logging

class PDFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(processed_dir: Path):
    """โหลดข้อมูลที่ประมวลผลแล้ว"""
    data_path = processed_dir / 'processed_documents.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    texts = [doc['text'] for doc in documents]
    # สมมติว่าเป็นงาน binary classification
    # ในสถานการณ์จริง คุณต้องกำหนดป้ายกำกับตามความเหมาะสม
    labels = [1 if len(doc['text']) > 200 else 0 for doc in documents]
    
    return texts, labels

def train_model(config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"ใช้งานอุปกรณ์: {device}")

    # โหลดข้อมูล
    processed_dir = Path(config['data']['processed_dir'])
    texts, labels = load_data(processed_dir)
    
    # แบ่งข้อมูล
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # เตรียมโมเดลและ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'], 
        num_labels=2
    ).to(device)

    # สร้าง datasets
    train_dataset = PDFDataset(train_texts, train_labels, tokenizer)
    val_dataset = PDFDataset(val_texts, val_labels, tokenizer)

    # สร้าง dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['model']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['model']['batch_size']
    )

    # เตรียม optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate'])
    )

    # เทรนโมเดล
    for epoch in range(config['model']['epochs']):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})

        # คำนวณ validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f'Epoch {epoch + 1}:')
        logging.info(f'Average training loss: {avg_train_loss:.4f}')
        logging.info(f'Average validation loss: {avg_val_loss:.4f}')

    # บันทึกโมเดล
    model_dir = Path(config['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'trained_model'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    logging.info(f'บันทึกโมเดลไปที่: {model_path}')

def main():
    config = load_config('config.yaml')
    setup_logging(config['log_dir'])
    
    try:
        train_model(config)
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาด: {str(e)}")
        raise

if __name__ == '__main__':
    main() 