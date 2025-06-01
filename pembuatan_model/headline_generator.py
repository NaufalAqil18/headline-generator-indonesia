import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import re
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, articles, titles, tokenizer, max_article_length=512, max_title_length=128):
        self.articles = articles
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_article_length = max_article_length
        self.max_title_length = max_title_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = str(self.articles[idx])
        title = str(self.titles[idx])

        # Tokenize article
        article_encoding = self.tokenizer(
            article,
            max_length=self.max_article_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize title
        title_encoding = self.tokenizer(
            title,
            max_length=self.max_title_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': article_encoding['input_ids'].squeeze(),
            'attention_mask': article_encoding['attention_mask'].squeeze(),
            'labels': title_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': title_encoding['attention_mask'].squeeze()
        }

class HeadlineGenerator:
    def __init__(self, model_name="cahya/bert2bert-indonesian-summarization", 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        
        # Create directories for saving artifacts
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.rouge_scores = []
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def prepare_data(self, df, batch_size=8, test_size=0.1, val_size=0.1, random_state=42):
        """Prepare train, validation, and test datasets"""
        # First split: training + validation vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Second split: training vs validation
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size/(1-test_size),
            random_state=random_state
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        train_dataset = NewsDataset(
            train_df['content'].values,
            train_df['title'].values,
            self.tokenizer
        )
        val_dataset = NewsDataset(
            val_df['content'].values,
            val_df['title'].values,
            self.tokenizer
        )
        test_dataset = NewsDataset(
            test_df['content'].values,
            test_df['title'].values,
            self.tokenizer
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader

    def compute_rouge_scores(self, predictions, references):
        """Compute ROUGE scores between predictions and references"""
        scores = {
            'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
            'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
            'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
        }
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            for metric in scores.keys():
                scores[metric]['precision'] += score[metric].precision
                scores[metric]['recall'] += score[metric].recall
                scores[metric]['fmeasure'] += score[metric].fmeasure
        
        # Average scores
        n = len(predictions)
        for metric in scores.keys():
            for key in scores[metric].keys():
                scores[metric][key] /= n
        
        return scores

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validating Epoch {epoch}'):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                decoder_attention_mask = batch['decoder_attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate predictions
                predictions = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode predictions and references
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                decoded_refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                all_predictions.extend(decoded_preds)
                all_references.extend(decoded_refs)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        rouge_scores = self.compute_rouge_scores(all_predictions, all_references)
        self.rouge_scores.append(rouge_scores)
        
        return avg_loss, rouge_scores

    def plot_training_progress(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot ROUGE scores
        plt.subplot(1, 2, 2)
        epochs = range(1, len(self.rouge_scores) + 1)
        
        plt.plot(epochs, [s['rouge1']['fmeasure'] for s in self.rouge_scores], label='ROUGE-1')
        plt.plot(epochs, [s['rouge2']['fmeasure'] for s in self.rouge_scores], label='ROUGE-2')
        plt.plot(epochs, [s['rougeL']['fmeasure'] for s in self.rouge_scores], label='ROUGE-L')
        
        plt.title('ROUGE Scores')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_progress.png')
        plt.close()

    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """Training loop with monitoring"""
        logger.info(f"\nStarting training on {self.device}")
        logger.info(f"Number of epochs: {epochs}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch + 1)
            logger.info(f"Average training loss: {train_loss:.4f}")
            
            # Validation phase
            val_loss, rouge_scores = self.validate(val_loader, epoch + 1)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Log ROUGE scores
            logger.info("\nROUGE Scores:")
            for metric, scores in rouge_scores.items():
                logger.info(f"{metric}: {scores['fmeasure']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'models/best_model_epoch_{epoch + 1}')
                logger.info(f"New best model saved!")
            
            # Plot progress
            self.plot_training_progress()
            
            # Early stopping check
            if len(self.val_losses) > 2 and self.val_losses[-1] > self.val_losses[-2]:
                logger.warning("Validation loss increased. Consider early stopping.")
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    def save_model(self, path):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load model and tokenizer"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

    def generate_headline(self, article_text):
        """Generate a headline for a given article"""
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            article_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate headline
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode and return headline
        headline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return headline

def main():
    # Example usage
    model = HeadlineGenerator()
    
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')  # Replace with your dataset path
    
    # Prepare data
    train_loader, val_loader, test_loader = model.prepare_data(df)
    
    # Train model
    model.train(train_loader, val_loader, epochs=3)
    
    # Example headline generation
    article = """
    Pemerintah Indonesia resmi mengumumkan peluncuran program baru yang bertujuan untuk meningkatkan literasi digital di kalangan pelajar. 
    Program ini akan dilaksanakan di lebih dari 500 sekolah di seluruh provinsi mulai bulan depan, dengan pelatihan bagi guru dan penyediaan perangkat teknologi.
    """
    
    headline = model.generate_headline(article)
    print(f"\nGenerated headline: {headline}")

if __name__ == "__main__":
    main() 