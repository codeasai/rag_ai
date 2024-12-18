﻿# RAG (Retrieval-Augmented Generation) Pipeline Project

ระบบตอบคำถามอัตโนมัติโดยใช้เทคนิค RAG ที่ผสมผสานการค้นคืนเอกสารและการสร้างคำตอบ

## โครงสร้างโปรเจค


```mermaid
graph TD
A[Project Root] --> B[src/]
A --> C[data/]
A --> D[config/]
A --> E[models/]
A --> F[logs/]
B --> B1[prepare_rag.py]
B --> B2[train.py]
B --> B3[evaluate.py]
B --> B4[utils.py]
C --> C1[raw_pdfs/]
C --> C2[processed/]
```

```mermaid
graph TD
A[Start] --> B[Load Configuration]
B --> C[Prepare Data]
C --> D[Train Model]
D --> E[Evaluate Model]
E --> F[RAG Pipeline]
F --> G[End]
subgraph Config
B1[config.yaml] --> B
end
subgraph Data Preparation
C1[PDF Files] --> C2[Process Documents]
C2 --> C3[Create Embeddings]
C3 --> C4[Build Vector Store]
end
subgraph Training
D1[Split Data] --> D2[Train Model]
D2 --> D3[Save Model]
end
subgraph Evaluation
E1[Load Test Data] --> E2[Make Predictions]
E2 --> E3[Calculate Metrics]
end
subgraph RAG
F1[User Query] --> F2[Retrieve Documents]
F2 --> F3[Generate Answer]
F3 --> F4[Return Response]
end
```
```mermaid
flowchart LR
A[PDF Files] --> B[Text Extraction]
B --> C[Text Splitting]
C --> D[Create Embeddings]
D --> E[FAISS Vector Store]
```
```mermaid
sequenceDiagram
participant Config
participant DataLoader
participant Model
participant Optimizer
participant Storage
Config->>DataLoader: Load Training Parameters
DataLoader->>Model: Prepare Batches
loop Training Epochs
Model->>Model: Forward Pass
Model->>Optimizer: Calculate Loss
Optimizer->>Model: Update Weights
Model->>Storage: Save Checkpoints
end
Model->>Storage: Save Final Model
```
```mermaid
flowchart TD
A[User Query] --> B{Vector Store}
B --> C[Retrieve Similar Documents]
C --> D[Language Model]
D --> E[Generate Answer]
E --> F[Format Response]
```
bash
git clone https://github.com/codeasai/rag_ai.git
cd rag_ai
bash
pip install -r requirements.txt
bash
python src/prepare_rag.py
bash
python src/train.py
bash
python src/evaluate.py
yaml
data:
pdf_dir: "data/raw_pdfs"
processed_dir: "data/processed"
model:
name: "bert-base-multilingual-cased"
epochs: 3
batch_size: 16
training:
learning_rate: 2e-5

