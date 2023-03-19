from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import os.path

import pandas as pd
import numpy as np
import json
from datasets import load_dataset, DatasetDict, Dataset

from transformers import RobertaTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import Trainer

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, add_special_tokens = True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YU_DATA_PATH = '../reference/counter_context/data'
RETRAIN = True
NUM_LABEL = 3

def plot_confusion_matrix(y_preds, y_true, title):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title(title)
    plt.savefig(title)
    
def main():
    print("------------------- LOADING DATA -------------------")
    train_encoded = torch.load("train_encoded.pt")
    val_encoded = torch.load("val_encoded.pt")
    
    print("------------------- LOADING MODEL -------------------")
    model = AutoModelForSequenceClassification.from_pretrained("./roberta-base-finetune-2-class/checkpoint-2000")
    model.eval()

    batch_size = 8
    logging_steps = len(train_encoded) // batch_size
    
    model_ckpt = "roberta-base"
    model_name = model_ckpt + "-finetune-eval"
    
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=8,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 2)

    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=train_encoded,
                    eval_dataset=val_encoded)
    
    print("------------------- EVALUATING -------------------")
    val_pred = trainer.predict(val_encoded)
    
    print(val_pred)
    
    y_preds = np.argmax(val_pred.predictions, axis=1)
    
    # torch.save(y_preds, "y_preds.pt")
    y_valid = np.array(val_encoded["label"])
    
    plot_confusion_matrix(y_preds, y_valid, title="Confusion Matrix")
    

if __name__ == "__main__":
    main()
    