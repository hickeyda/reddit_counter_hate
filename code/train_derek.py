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

import torchmetrics as tm

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
    sent = item.context
    sent_pair = item.target
    attributes = F.one_hot(torch.tensor(item.label), num_classes=2).squeeze().type(torch.FloatTensor) # CHANGE CLASS IF DOING MULTILABEL HERE
    tokens = self.tokenizer.encode_plus(sent, sent_pair,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=self.max_token_len,
                                        padding='max_length',
                                        return_attention_mask=True)
    
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}


class HS_Data_Module(pl.LightningDataModule):
  def __init__(self, train_data, val_data, test_data, batch_size = 16, max_token_len=512, model_name='roberta_base'):
    super().__init__()
    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data
    self.batch_size = batch_size
    self.max_token_len = max_token_len
    self.model_name = model_name
    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

  def setup(self, stage = None):
    if stage in (None, 'fit'):
      self.train_dataset = HS_Dataset(self.train_data, self.tokenizer, sample_size = None)
      self.val_dataset = HS_Dataset(self.val_data, self.tokenizer, sample_size = None)
    if stage == 'predict':
      self.test_dataset = HS_Dataset(self.test_data, self.tokenizer, sample_size = None)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12, shuffle=False)


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
    self.log('Train Loss', loss, prog_bar = True, logger=True)

    # Get Train Accuracy
    self.train_acc.update(logits, batch['labels'])
    self.log('Train Acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # Get Train F1
    self.train_f1.update(logits, batch['labels'])
    self.log('Train F1', self.train_f1, on_step=True, on_epoch=True, logger=True)

    # Get Train AUROC
    self.train_auroc.update(logits, batch['labels'])
    self.log('Train AUROC', self.train_auroc, on_step=True, on_epoch=True, logger=True)

    return {"loss": loss, "predictions": logits, "labels": batch['labels']}

  def validation_step(self, batch, batch_idx):
    loss, logits = self(**batch)
    self.log('Val Loss', loss, prog_bar = True, logger=True)

    # Get Val Accuracy
    self.val_acc.update(logits, batch['labels'])
    self.log('Val Acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # Get Val F1
    self.val_f1.update(logits, batch['labels'])
    self.log('Val F1', self.val_f1, on_step=True, on_epoch=True, logger=True)

    # Get Val AUROC
    self.val_auroc.update(logits, batch['labels'])
    self.log('Val AUROC', self.val_auroc, on_step=True, on_epoch=True, logger=True)

    return {"loss": loss, "predictions": logits, "labels": batch['labels']}

  def predict_step(self, batch, batch_idx):
    _, logits = self(**batch)
    return logits

  def configure_optimizers(self):
    optimzer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
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

    # Set batch size (needed for data module and config)
    b_size = 64

    # Create data set
    hs_data_module = HS_Data_Module(train_df, val_df, test_df, batch_size = b_size)
    hs_data_module.setup()

    # Config
    config = {
        'model_name': 'roberta-base',
        'n_labels': 2,
        'batch_size': b_size,
        'lr': 1.5e-6,
        'w_decay': 0.001,
        'train_size': len(hs_data_module.train_dataloader()),
        'warmup': 0.2,
        'n_epochs': 10
    }

    # Model
    model = HS_Classifier(config)

    # Early Stopping (based on validation loss not decreasing)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='Val Loss', patience=5, min_delta=0.0005, verbose=True, mode='min')

    # Train
    trainer = pl.Trainer(max_epochs=config['n_epochs'],
                        callbacks=[early_stop_callback],
                        num_sanity_val_steps=50)

    trainer.fit(model, hs_data_module)

    y_pred_probs = get_preds(model, hs_data_module)
    y_preds = torch.argmax(y_pred_probs, 1).numpy()
    y_true = test_df.label.values

    return

if __name__ == "__main__":
    main() # CODE MAINLY COMES FROM: https://www.youtube.com/watch?v=vNKIg8rXK6w
