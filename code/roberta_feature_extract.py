from preprocess_data import yuetal_data_preprocess
from datasets import Dataset
import pandas as pd
import torch
import numpy as np

from transformers import AutoModel, RobertaTokenizer

from sklearn.linear_model import LogisticRegression

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YU_DATA_PATH = '../reference/counter_context/data'
RETRAIN = False

model_ckpt = "roberta-base"
model = AutoModel.from_pretrained("roberta-base", num_labels=2).to(DEV)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(DEV) for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def tokenize(batch):
    return tokenizer(batch["text"], 
                     padding=True, 
                     truncation=True, 
                     add_special_tokens = True)

def lr_classifier(train_hidden, val_hidden):
    X_train = np.array(train_hidden["hidden_state"])
    X_valid = np.array(val_hidden["hidden_state"])
    y_train = np.array(train_hidden["label"])
    y_valid = np.array(val_hidden["label"])
    X_train.shape, X_valid.shape

    lr_clf = LogisticRegression(max_iter=10000, verbose=1)
    lr_clf.fit(X_train, y_train)

    print("Validation Accuracy = ", lr_clf.score(X_valid, y_valid))

def main():
    if RETRAIN:
        train_df = yuetal_data_preprocess(YU_DATA_PATH + '/gold/train.jsonl', 
                                        YU_DATA_PATH + '/silver/train.jsonl')
        val_df = yuetal_data_preprocess(YU_DATA_PATH + '/gold/val.jsonl', 
                                        YU_DATA_PATH + '/silver/val.jsonl')

        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        
        
        train_encoded = train_ds.map(tokenize, batched=True, batch_size=None)
        val_encoded = val_ds.map(tokenize, batched=True, batch_size=None)
        
        train_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
        val_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                                "label"])
        
        train_hidden = train_encoded.map(extract_hidden_states, batched=True, batch_size=16)
        val_hidden = val_encoded.map(extract_hidden_states, batched=True, batch_size=16)

        # Save the hidden state
        torch.save(train_hidden, "train_hidden.pt")
        torch.save(val_hidden, "val_hidden.pt")
    else:
        train_hidden = torch.load("train_hidden.pt")
        val_hidden = torch.load("val_hidden.pt")

    print("------------------- Training -------------------")
    lr_classifier(train_hidden, val_hidden)

if __name__ == "__main__":
    main()
    