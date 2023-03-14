# !pip install transformers
# !pip install pytorch-lightning
# COLAB NOTEBOOK: https://colab.research.google.com/drive/17EbBEPdCz25C2jty8kHFIGOEeBZ0NARN?usp=sharing

import pandas as pd
from transformers import RobertaTokenizer
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import os
import json
import torch.nn.functional as F

import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_path = 'LINK_TO_DATA_PATH'

def load_data_file(path):
    lines_list = []
    with open(path, 'r') as f:
        for line in f:
            lines_list.append(json.loads(line))

    df = pd.DataFrame(lines_list)
    
    return df

def binarize_label(label):
    if int(label) == 1: #"Neutral" (1) label combines with hate (0) to form "Not Counter-Hate"
        label = 0
    if int(label) == 2: # Change "Counter-Hate" label from 2 to 1 (binary, needed for one hot encoding in HS_Dataset class)
      label = 1
    return int(label)


class HS_Dataset(Dataset):
  # Define data set with data, tokenizer, and max token length
  def __init__(self, data, tokenizer, max_token_len=256):
    self.data = data
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def __len__(self):
    return(len(self.data))

  def __getitem__(self, index):
    item = self.data.iloc[index]
    sent = str(item.context + ' [SEP] ' + item.target)
    attributes = F.one_hot(torch.tensor(item.label), num_classes=2).squeeze().type(torch.FloatTensor)
    tokens = self.tokenizer.encode_plus(sent,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=self.max_token_len,
                                        padding='max_length',
                                        return_attention_mask=True)
    
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}


class HS_Data_Module(pl.LightningDataModule):
  def __init__(self, train_data, val_data, batch_size = 16, max_token_len=512, model_name='roberta_base'):
    super().__init__()
    self.train_data = train_data
    self.val_data = val_data
    self.batch_size = batch_size
    self.max_token_len = max_token_len
    self.model_name = model_name
    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

  def setup(self, stage = None):
    if stage in (None, 'fit'):
      self.train_dataset = HS_Dataset(self.train_data, self.tokenizer)
      self.val_dataset = HS_Dataset(self.val_data, self.tokenizer)
    if stage == 'predict': # CHANGE TO TEST DATA
      self.val_dataset = HS_Dataset(self.val_data, self.tokenizer)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)

  def predict_dataloader(self): # CHANGE TO TEST DATA
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)


class HS_Classifier(pl.LightningModule):
  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
    self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classification = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    torch.nn.init.xavier_uniform_(self.hidden.weight)
    torch.nn.init.xavier_uniform_(self.classification.weight)
    self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean') # combines loss and sigmoid
    self.dropout = nn.Dropout()

  def forward(self, input_ids, attention_mask, labels=None):
    out = self.pretrained_model(input_ids=input_ids, attention_mask = attention_mask) # model output
    pooled_out = torch.mean(out.last_hidden_state, 1) # take mean output 
    pooled_out = self.hidden(pooled_out)
    pooled_out = self.dropout(pooled_out)
    pooled_out = F.relu(pooled_out)
    logits = self.classification(pooled_out)

    loss = 0
    if labels is not None:
      loss = self.loss_fn(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
    return loss, logits

  def training_step(self, batch, batch_idx):
    loss, logits = self(**batch)
    self.log('train loss', loss, prog_bar = True, logger=True)
    return {"loss": loss, "predictions": logits, "labels": batch['labels']}

  def validation_step(self, batch, batch_idx):
    loss, logits = self(**batch)
    self.log('val loss', loss, prog_bar = True, logger=True)
    return {"loss": loss, "predictions": logits, "labels": batch['labels']}

  def predict_step(self, batch, batch_idx):
    _, logits = self(**batch)
    return logits

  def configure_optimizers(self):
    optimzer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
    total_steps = self.config['train_size'] / self.config['batch_size']
    warmup_steps = math.floor(total_steps *  self.config['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimzer, warmup_steps, total_steps)
    return [optimzer], [scheduler]

def get_preds(model, dm):
  preds = trainer.predict(model, datamodule=dm)
  flat_preds = torch.stack([torch.sigmoid(torch.Tensor(p)) for batch in preds for p in batch])
  return flat_preds

def main():
    #"Gold" and "Silver" directories refer to annotated data with different levels of inter-annotator agreement
    train_gold_df = load_data_file(data_path + '/gold/train.jsonl')
    val_gold_df = load_data_file(data_path + '/gold/val.jsonl')
    test_df = load_data_file(data_path + '/gold/test.jsonl') # The test set only comes from the 'gold' category

    train_silver_df = load_data_file(data_path + '/silver/train.jsonl')
    val_silver_df = load_data_file(data_path + '/silver/val.jsonl')

    #Combine Gold and Silver
    train_df = pd.concat([train_gold_df, train_silver_df])
    train_df['label'] = train_df['label'].apply(binarize_label).values
    val_df = pd.concat([val_gold_df, val_silver_df])
    val_df['label'] = val_df['label'].apply(binarize_label).values

    # Create data set
    hs_data_module = HS_Data_Module(train_df, val_df, batch_size = 64) # CHANGE BATCH SIZE HERE AND IN CONFIG
    hs_data_module.setup()

    # Config
    config = {
        'model_name': 'roberta-base',
        'n_labels': 2,
        'batch_size': 64,
        'lr': 1.5e-6,
        'w_decay': 0.001,
        'train_size': len(hs_data_module.train_dataloader()),
        'warmup': 0.2,
        'n_epochs': 10
    }

    # Model
    model = HS_Classifier(config)

    # Train
    trainer = pl.Trainer(max_epochs=config['n_epochs'], 
                        gpus=1, 
                        num_sanity_val_steps=50)

    trainer.fit(model, hs_data_module)

    v_preds = get_preds(model, hs_data_module)
    val_y = [int(x) for x in val_df['label'].values]
    val_preds = torch.argmax(v_preds, dim=1).numpy()
    print("Val Accuracy:", accuracy_score(val_y, val_preds))
    print("Val F1:", f1_score(val_y, val_preds, average='weighted'))
    print("Val Recall:", recall_score(val_y, val_preds, average='weighted'))
    print("Val Precision:", precision_score(val_y, val_preds, average='weighted'))

if __name__ == "__main__":
    main() # CODE MAINLY COMES FROM: https://www.youtube.com/watch?v=vNKIg8rXK6w
