"""
predict.py
Load the fine-tuned model and provide a prediction function.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_dir="./saved_distilbert", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        # label mapping - adjust if you used different mapping
        self.id2label = {0: "negative", 1: "positive"}

    def predict(self, text, max_len=128):
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
            pred_id = int(np.argmax(logits))
        return {
            "label": self.id2label[pred_id],
            "score": float(probs[pred_id]),
            "probs": probs.tolist()
        }

if __name__ == "__main__":
    sa = SentimentAnalyzer("./saved_distilbert")
    examples = [
        "I love this product! Works exactly as expected.",
        "Terrible. Broke after two days and customer service ignored me.",
        "Not bad for the price, but could be better."
    ]
    for ex in examples:
        print(ex)
        print(sa.predict(ex))
        print("-" * 40)
