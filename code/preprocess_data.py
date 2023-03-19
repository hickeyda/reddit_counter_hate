from datasets import Dataset
from transformers import AutoModel, RobertaTokenizer

import pandas as pd
import torch
import numpy as np
import os
import json

import torch

from sklearn.utils import resample

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "roberta-base"
TOKENIZER = RobertaTokenizer.from_pretrained(MODEL_CKPT)
BASE_MODEL = AutoModel.from_pretrained(MODEL_CKPT).to(DEV)
YU_DATA_PATH = '../reference/counter_context/data'

def load_data_file(path):
    lines_list = []
    with open(path, 'r') as f:
        for line in f:
            lines_list.append(json.loads(line))

    df = pd.DataFrame(lines_list)
    
    return df

def upsample(df):
    # Separate majority and minority classes
    majority_df = df[df['label']==0]
    minority_df = df[df['label']==1]
    
    # Upsample minority class
    upsampled_minority_df = resample(minority_df, 
                                    replace=True,     # sample with replacement
                                    n_samples=len(majority_df),    # to match majority class
                                    random_state=123) # reproducible results
    
    return pd.concat([majority_df, upsampled_minority_df])

def yuetal_data_preprocess(gold_data_path, silver_data_path):
    """
    gold_data_path: path to gold jsonl data file
    silver_data_path: path to silver jsonl data file
    """
    gold_df = load_data_file(gold_data_path)
    silver_df = load_data_file(silver_data_path)
    
    df = pd.concat([gold_df, silver_df])
    
    # Binarize and cast label to int
    df['label'] = list(map(lambda x: 0 if int(x) == 1 or int(x) == 0 else 1, df['label']))
    
    return df

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(DEV) for k,v in batch.items() 
              if k in TOKENIZER.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = BASE_MODEL(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def tokenize(batch):
    return TOKENIZER(text=batch["target"],
                       text_pair=batch["context"],
                       add_special_tokens=True, 
                       truncation=True, 
                       max_length=512, 
                       padding='max_length', 
                       return_attention_mask=True)


def main():
    if not os.path.exists("./train_hidden.pt") or not os.path.exists("./val_hidden.pt"):
        train_df = yuetal_data_preprocess(YU_DATA_PATH + '/gold/train.jsonl', 
                                        YU_DATA_PATH + '/silver/train.jsonl')
        val_df = yuetal_data_preprocess(YU_DATA_PATH + '/gold/val.jsonl', 
                                        YU_DATA_PATH + '/silver/val.jsonl')

        train_df = upsample(train_df)
        
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        
        train_encoded = train_ds.map(tokenize, batched=True, batch_size=None)
        val_encoded = val_ds.map(tokenize, batched=True, batch_size=None)
        
        # Save the hidden state
        torch.save(train_encoded, "train_encoded.pt")
        torch.save(val_encoded, "val_encoded.pt")
        
        train_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                                "label"])
        val_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                                "label"])
        
        train_hidden = train_encoded.map(extract_hidden_states, batched=True, batch_size=32)
        val_hidden = val_encoded.map(extract_hidden_states, batched=True, batch_size=32)

        # Save the hidden state
        torch.save(train_hidden, "train_hidden.pt")
        torch.save(val_hidden, "val_hidden.pt")

if __name__ == "__main__":
    main()
    