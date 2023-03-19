import os

import torch
import pandas as pd

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "roberta-base"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    print("------------------- Data Loading -------------------")
    train_encoded = torch.load("train_encoded.pt")
    val_encoded = torch.load("val_encoded.pt")
    
    train_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    val_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=2)

    batch_size = 8
    logging_steps = len(train_encoded) // batch_size
    model_name = MODEL_CKPT + "-finetune-2-class"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=10,
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
    
    print("------------------- Training -------------------")
    trainer.train()

if __name__ == "__main__":
    main()
    