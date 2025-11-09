"""
Fine-tune distilbert-base-uncased on a reviews dataset.
Expected CSV with columns: "review", "label"
Where label is 0 (negative) or 1 (positive) (or you can map more classes).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load CSV
    
    df=pd.read_csv('/Users/rushikeshiname/RushiIname/Python/uploads/tripadvisor_hotel_reviews.csv')
    df['label'] = df['Rating'].apply(lambda x: 1 if x > 3 else 0)
    df['review']=df['Review']
    if 'review' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'label' columns")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].tolist(), df['label'].tolist(),
        test_size=args.val_split, random_state=42, stratify=df['label'].tolist()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = ReviewsDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
    val_ds = ReviewsDataset(val_texts, val_labels, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        total_loss = 0.0
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                preds.append(logits)
                true.append(label_ids)
        preds = np.vstack(preds)
        true = np.concatenate(true)
        metrics = compute_metrics(preds, true)
        print(f"Validation metrics: {metrics}")

        # Save best
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best model (f1={best_f1:.4f}) to {args.output_dir}")

    print("Training complete. Best F1:", best_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="reviews.csv", help="Path to CSV file with 'review' and 'label' columns")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./saved_distilbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_labels", type=int, default=2)
    args = parser.parse_args()
    train(args)
