import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import argparse


class IntentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        df = pd.read_csv(csv_path)
        self.texts = df["Text"].tolist()
        self.labels = df["Answer"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",   # pad short sequences to max_length
            truncation=True,        # cut sequences longer than max_length
            return_tensors="pt",    # return PyTorch tensors, not plain lists
        )
        return {
            # squeeze() removes the extra batch dimension the tokenizer adds
            # shape goes from [1, 256] → [256]
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            # float32 tensor containing the soft label probability
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }



class IntentClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # bert_output.last_hidden_state shape: [batch, seq_len, 768]
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        # [CLS] token is always at position 0
        cls_vector = bert_output.last_hidden_state[:, 0, :]  # [batch, 768]
        cls_vector = self.dropout(cls_vector)
        logit = self.classifier(cls_vector)                  # [batch, 1]
        return logit.squeeze(1)                              # [batch]



def kl_loss(logits, soft_labels):
    # Build 2-class target distribution from scalar soft labels
    # shape: [batch] → [batch, 2]
    targets = torch.stack([1 - soft_labels, soft_labels], dim=1)

    # Build 2-class predicted distribution from logits
    probs = torch.sigmoid(logits)
    preds = torch.stack([1 - probs, probs], dim=1)
    log_preds = torch.log(preds + 1e-8)  # +1e-8 avoids log(0)

    return F.kl_div(log_preds, targets, reduction="batchmean")



def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss, correct, tp, fp, fn = 0.0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = kl_loss(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= threshold).long()
            hard_lbls = (labels >= threshold).long()

            correct += (preds == hard_lbls).sum().item()
            tp += ((preds == 1) & (hard_lbls == 1)).sum().item()
            fp += ((preds == 1) & (hard_lbls == 0)).sum().item()
            fn += ((preds == 0) & (hard_lbls == 1)).sum().item()

    n = len(loader.dataset)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "loss":      total_loss / len(loader),
        "accuracy":  correct / n,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = IntentDataset("assets/train.csv", tokenizer)
    val_dataset = IntentDataset("assets/val.csv",   tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,   batch_size=args.batch_size)

    model = IntentClassifier(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = kl_loss(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.3f} | "
            f"val_f1: {val_metrics['f1']:.3f} | "
            f"precision: {val_metrics['precision']:.3f} | "
            f"recall: {val_metrics['recall']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), "assets/best_model.pt")
            print(f"  → Saved new best model (val_loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT intent classifier")
    parser.add_argument("--model",      default="bert-base-uncased")
    parser.add_argument("--epochs",     type=int,   default=6)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
