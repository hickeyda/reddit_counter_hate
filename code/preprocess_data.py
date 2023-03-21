from datasets import Dataset
from transformers import AutoModel, RobertaTokenizer

import pandas as pd
import torch
import numpy as np
import os
import json
import random

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

def data_preprocess(data_set, file_name, 
                    upsampling=False, 
                    num_label=2, 
                    output_hidden_states=True,
                    overwrite=False,
                    file_type='jsonl',
                    null_ratio=0.0):
    """
    gold_data_path: path to gold jsonl data file
    silver_data_path: path to silver jsonl data file
    """
    new_file_name = file_name
    
    df = load_data_file(YU_DATA_PATH + "/" + data_set + "/" + file_name + "." + file_type)
    
    # Binarize and cast label to int
    if num_label == 2:
        new_file_name = "2-label_" + new_file_name
        df['label'] = list(map(lambda x: 0 if int(x) == 1 or int(x) == 0 else 1, df['label']))
    else:
        new_file_name = "3-label_" + new_file_name
        df['label'] = list(map(lambda x: int(x), df['label']))
    
    if upsampling:
        new_file_name = "upsample_" + new_file_name
        df = upsample(df)
    
    # Inject null contexts
    if null_ratio != 0.0:
        new_file_name = "null_context_" + new_file_name
        null_indices = random.sample(list(range(len(df))), int(null_ratio * len(df)))
        df['context'].update(pd.Series("", index=null_indices))
    
    # Tokenize
    ds = Dataset.from_pandas(df)
    tokens = ds.map(tokenize, batched=True, batch_size=None)
    tokens_fn = data_set + '_' + new_file_name + "_tokens.pt"
    if not os.path.exists(tokens_fn) or overwrite == True:
        torch.save(tokens, tokens_fn)
    
    # Extract hidden states
    if output_hidden_states:
        tokens.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        hiddens = tokens.map(extract_hidden_states, batched=True, batch_size=32)
        torch.save(hiddens, data_set + '_' + new_file_name + "_hidden.pt")
        hiddens_fn = data_set + '_' + new_file_name + "_tokens.pt"
        if not os.path.exists(hiddens_fn) and overwrite == True:
            torch.save(hiddens, hiddens_fn)

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
    data_preprocess(data_set="gold", file_name="train", upsampling=False, num_label=2)
    data_preprocess(data_set="gold", file_name="train", upsampling=True, num_label=2)
    data_preprocess(data_set="gold", file_name="val", upsampling=False, num_label=2)
    data_preprocess(data_set="silver", file_name="train", upsampling=False, num_label=2)
    data_preprocess(data_set="silver", file_name="train", upsampling=True, num_label=2)
    data_preprocess(data_set="silver", file_name="val", upsampling=False, num_label=2)
    
    data_preprocess(data_set="gold", file_name="train", upsampling=False, num_label=3, null_ratio=0.25, output_hidden_states=False)
    data_preprocess(data_set="gold", file_name="val", upsampling=False, num_label=3, null_ratio=0.25, output_hidden_states=False)
    data_preprocess(data_set="silver", file_name="train", upsampling=False, num_label=3, null_ratio=0.25, output_hidden_states=False)
    data_preprocess(data_set="silver", file_name="val", upsampling=False, num_label=3, null_ratio=0.25, output_hidden_states=False)
    
if __name__ == "__main__":
    main()
    