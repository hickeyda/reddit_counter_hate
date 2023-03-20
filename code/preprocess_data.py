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

YU_DATA_PATH = '../reference/counter_context/data'
def data_preprocess(data_set, file_name, file_type, upsampling):
    """
    gold_data_path: path to gold jsonl data file
    silver_data_path: path to silver jsonl data file
    """
    df = load_data_file(YU_DATA_PATH + "/" + data_set + "/" + file_name + "." + file_type)
    
    if upsampling:
        file_name = "upsample_" + file_name
    
    # Binarize and cast label to int
    df['label'] = list(map(lambda x: 0 if int(x) == 1 or int(x) == 0 else 1, df['label']))
    
    if upsampling:
        df = upsample(df)
    
    # Tokenize
    ds = Dataset.from_pandas(df)
    tokens = ds.map(tokenize, batched=True, batch_size=None)
    torch.save(tokens, data_set + '_' + file_name + "_tokens.pt")
    
    # Extract hidden states
    tokens.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    hiddens = tokens.map(extract_hidden_states, batched=True, batch_size=32)
    torch.save(hiddens, data_set + '_' + file_name + "_hidden.pt")

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
    settings = [
        ("gold", "train", False),   # data set, file name, upsample
        ("gold", "train", True),
        ("gold", "val", False),
        ("silver", "train", False),
        ("silver", "train", True),
        ("silver", "val", False),
    ]
    
    file_names = []
    for setting in settings:
        fn = setting[0] + "_" + setting[1]
        if setting[2]:
            fn = "upsample_" + fn
        file_names.append(fn + "_hidden.pt")
        file_names.append(fn + "_tokens.pt")
        
    for fn in file_names:
        if not os.path.exists(fn):
            for setting in settings:
                data_preprocess(setting[0], setting[1], "jsonl", setting[2])
            break
    
if __name__ == "__main__":
    main()
    