import os

import torch
import pandas as pd

from datasets import concatenate_datasets

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "MODEL" #Use Derek's target only model trained on just gold
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

def predict_and_add_to_train(model, trainer, train_set, unlabeled_data, threshold):
    converged = False
    convergence_limit = 50

    model.eval()

    unlabeled_pred = trainer.predict(unlabeled_data) # Make predictions for unlabeled data

    train_idx = np.where(np.max(unlabeled_pred, axis=1) > threshold)[0] # get indices of high confidence predictions
    unlabeled_idx = np.where(np.max(unlabeled_pred, axis=1) <= threshold)[0] # get indices of not high confidence predictions

    labels = np.argmax(unlabeled_pred, axis=1) #label 0 or 1

    train_labels = labels[train_idx] #Specify we want labels of new train data

    new_unlabeled = unlabeled_data[unlabeled_idx] #data that is still unlabeled

    add_to_train_set = (unlabeled_data[train_idx], train_labels) # make a dataset of labeled data to add to training (not sure exact command)

    new_unlabeled = torch.stack(new_unlabeled) # Create a new unlabeled dataset that can be evaluated
    train_set.append(add_to_train_set) # Add the new training data. I'm not sure the exact command for this

    if len(train_labels) < convergence_limit: # if only a certain number of predictions are added to training data
        converged = True

    print(f"{len(train_labels)} points added to training, {len(new_unlabeled)} remaining")

    return new_unlabeled, train_set, converged
    
def main():
    print("------------------- Data Loading -------------------")
    train_encoded = torch.load("train_encoded.pt") #use just gold data, binary label
    val_encoded = torch.load("val_encoded.pt") #use just gold data, binary label
    unlabeled_encoded = torch.load(UNLABELED_DATA_PATH) #load in unlabeled data
    
    train_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    val_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=2)

    batch_size = 64
    logging_steps = len(train_encoded) // batch_size
    model_name = MODEL_CKPT + "-finetune-2-class"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=3, #set epochs to something low since you'll be training a lot? maybe?
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 1)

    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=train_encoded,
                    eval_dataset=val_encoded)
    
    converged = False
    
    print("------------------- Training -------------------")

    i = 0

    unlabeled_encoded, train_encoded, converged = predict_and_add_to_train(model, trainer, train_encoded, unlabeled_encoded, 0.8)
    

    while not converged:
        print("Step", i)

        trainer = Trainer(model=model, args=training_args, #update trainer with new data
                compute_metrics=compute_metrics,
                train_dataset=train_encoded,
                eval_dataset=val_encoded)
        trainer.train()

        #trainer.save_model(f'./self_train_{i}')

        #model = AutoModelForSequenceClassification.from_pretrained(f'./self_train_{i}', 
        #                                                    num_labels=2)
        
        unlabeled_encoded, train_encoded, converged = predict_and_add_to_train(model, trainer, train_encoded, unlabeled_encoded, 0.8)

        i += 1

    

if __name__ == "__main__":
    main()