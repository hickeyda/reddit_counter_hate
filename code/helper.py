import pandas as pd
import json

import torch

from transformers import AutoModel, RobertaTokenizer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "roberta-base"
MODEL = AutoModel.from_pretrained(MODEL_CKPT, num_labels=2).to(DEV)
TOKENIZER = RobertaTokenizer.from_pretrained(MODEL_CKPT)

def load_data_file(path):
    lines_list = []
    with open(path, 'r') as f:
        for line in f:
            lines_list.append(json.loads(line))

    df = pd.DataFrame(lines_list)
    
    return df

def yuetal_data_preprocess(gold_data_path, silver_data_path):
    """
    gold_data_path: path to gold jsonl data file
    silver_data_path: path to silver jsonl data file
    """
    gold_df = load_data_file(gold_data_path)
    silver_df = load_data_file(silver_data_path)
    
    df = pd.concat([gold_df, silver_df])
    
    # Combine context and target
    contexts = df['context'].values
    targets = df['target'].values
    df['text'] = list(map(lambda x,y: str(x) + ' [SEP] ' + str(y), contexts, targets))
    df = df.drop(columns=['context', 'target'])
    
    # Binarize and cast label to int
    df['label'] = list(map(lambda x: 0 if int(x) == 1 else int(x), df['label']))
    
    return df

def extract_hidden_states(batch):
    # Place MODEL inputs on the GPU
    inputs = {k:v.to(DEV) for k,v in batch.items() 
              if k in TOKENIZER.model_input_names}
    
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = MODEL(**inputs).last_hidden_state
        
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
