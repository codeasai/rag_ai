import os
import torch
from pathlib import Path
from src.loader import PDFLoader
from src.processor import TextProcessor
from src.trainer import QATrainer
from src.utils import load_config, setup_logging, create_directories
import logging


class PDFAISystem:
    def __init__(self, config_path="configs/config.yaml"):
        # Load configuration
        self.config = load_config(config_path)

        # Setup directories
        create_directories(self.config)
        setup_logging(self.config['log_dir'])
        self.logger = logging.getLogger(__name__)

        # Check CUDA availability
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.logger.info("GPU available - using CUDA")
            self.device = 'cuda'
        else:
            self.logger.info("No GPU detected - using CPU")

        # Initialize components
        self.loader = PDFLoader(self.config['data']['pdf_dir'])
        self.processor = TextProcessor(
            chunk_size=self.config['data']['chunk_size'],
            chunk_overlap=self.config['data']['chunk_overlap']
        )
        self.trainer = QATrainer(
            model_name=self.config['model']['name'],
            device=self.device
        )

    def run(self):
        try:
            # Step 1: Load PDFs
            self.logger.info("Loading PDF documents...")
            documents = self.loader.load_all_pdfs()
            self.logger.info(f"Loaded {len(documents)} documents")

            # Step 2: Process text
            self.logger.info("Processing documents...")
            chunks, vectors = self.processor.process_documents(
                documents,
                self.config['data']['processed_dir']
            )
            self.logger.info(f"Created {len(chunks)} chunks")

            # Step 3: Train model
            self.logger.info("Starting model training...")
            encoded_data, tokenizer = self.trainer.prepare_training_data(chunks)
            model = self.trainer.train(
                encoded_data,
                epochs=self.config['model']['epochs']
            )

            # Step 4: Save results
            self.logger.info("Saving model and tokenizer...")
            model_save_path = Path(self.config['model_dir']) / 'qa_model'
            tokenizer_save_path = Path(self.config['model_dir']) / 'tokenizer'

            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(tokenizer_save_path)

            self.logger.info("Training completed successfully!")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise


if __name__ == "__main__":
    # Create simple config if not exists
    if not os.path.exists("configs/config.yaml"):
        basic_config = {
            "data": {
                "pdf_dir": "data/raw_pdfs",
                "processed_dir": "data/processed",
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "model": {
                "name": "bert-base-uncased",
                "epochs": 1
            },
            "model_dir": "models",
            "log_dir": "logs"
        }
        os.makedirs("configs", exist_ok=True)
        with open("configs/config.yaml", "w") as f:
            import yaml

            yaml.dump(basic_config, f)

    # Run system
    system = PDFAISystem()
    system.run()
