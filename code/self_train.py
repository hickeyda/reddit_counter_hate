import torch
import pandas as pd

from datasets import concatenate_datasets, Dataset

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "model-4-base"
SEED = 123

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train_3_label_stance_base():
    print("------------------- Data Loading -------------------")
    gold_train_tokens = torch.load("gold_3-label_train_tokens.pt")
    silver_train_tokens = torch.load("silver_3-label_train_tokens.pt")
    
    gold_val_tokens = torch.load("gold_3-label_val_tokens.pt")
    silver_val_tokens = torch.load("silver_3-label_val_tokens.pt")
    
    mixed_train_tokens = concatenate_datasets([gold_train_tokens, silver_train_tokens]).shuffle(seed=SEED)
    mixed_val_tokens = concatenate_datasets([gold_val_tokens, silver_val_tokens]).shuffle(seed=SEED)
    
    gold_train_tokens.shuffle(seed=SEED)
    silver_train_tokens.shuffle(seed=SEED)
    
    gold_train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    gold_val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    mixed_train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    mixed_val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=3)
    batch_size = 4
    model_name = MODEL_CKPT + "-finetune-3-class"
    
    logging_steps = len(mixed_train_tokens) // batch_size
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
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
                    train_dataset=mixed_train_tokens,
                    eval_dataset=mixed_val_tokens)
    
    print("------------------- Training with mixed data -------------------")
    trainer.train()
    
    logging_steps = len(gold_train_tokens) // batch_size
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=7,
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
                    train_dataset=gold_train_tokens,
                    eval_dataset=gold_val_tokens)
    print("------------------- Training with gold data -------------------")
    trainer.train()

def predict_and_add_to_train(model, trainer, train_tokens, unlabeled_tokens, threshold):
    converged = False
    convergence_limit = 50

    model.eval()

    pred_logits = torch.Tensor(trainer.predict(unlabeled_tokens).predictions)
    pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
    max_probs = torch.max(pred_probs, dim=1).numpy()
    labels = np.argmax(max_probs, axis=1) #label 0 or 1

    # Update train tokens
    confident_pred_indices = np.where(max_probs > threshold)
    # new_unlabeled = torch.Tensor(unlabeled_tokens[~confident_pred_indices])
    
    pseudo_train_tokens = Dataset.from_dict(unlabeled_tokens[confident_pred_indices])
    pseudo_train_labels = labels[confident_pred_indices]
    pseudo_train_tokens.add_column("label", pseudo_train_labels)
    
    train_tokens = concatenate_datasets(train_tokens, pseudo_train_tokens)
    
    # Update unlabeled tokens
    remain_indices = []
    for i in range(len(unlabeled_tokens)):
        if i not in confident_pred_indices:
            remain_indices.append(i)
    
    unlabeled_tokens = Dataset.from_dict(unlabeled_tokens[remain_indices])
    
    # if only a certain number of predictions are added to training data
    if len(confident_pred_indices) < convergence_limit:
        converged = True

    print(f"{len(confident_pred_indices)} points added to training, {len(unlabeled_tokens)} remaining")

    return train_tokens, unlabeled_tokens, converged
    
def main():
    print("------------------- Data Loading -------------------")
    train_tokens = torch.load("gold_2-label_train_tokens.pt") #use just gold data, binary label
    val_tokens = torch.load("gold_2-label_val_tokens.pt") #use just gold data, binary label
    unlabeled_tokens = torch.load("combined_subreddit.pt") #load in unlabeled data
    
    train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    unlabeled_tokens.set_format("torch", columns=["input_ids", "attention_mask"])
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=2)

    batch_size = 8
    logging_steps = len(train_tokens) // batch_size
    model_name = MODEL_CKPT + "-self-train-2-class"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=1, #set epochs to something low since you'll be training a lot? maybe?
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_strategy="no")
    
    print()

    print("------------------- Training -------------------")
    i = 0
    converged = False
    while not converged:
        print("Step", i)

        trainer = Trainer(model=model, args=training_args, #update trainer with new data
                compute_metrics=compute_metrics,
                train_dataset=train_tokens,
                eval_dataset=val_tokens)
        trainer.train()
        
        train_tokens, unlabeled_tokens, converged = \
            predict_and_add_to_train(model, trainer, train_tokens, unlabeled_tokens, 0.8)

        i += 1
        
    trainer.save_model("self_train")

    

if __name__ == "__main__":
    main()