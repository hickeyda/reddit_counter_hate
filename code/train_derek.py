path = "..." # path to base directy
YU_DATA_PATH = path + 'data/' # set path to folder holding data
feat_path = path + "encoded/" # set path to folder holding encoded data

from shift_correction import *

import pandas as pd
import numpy as np
import json
import os

from datasets import Dataset

import torch
from transformers import AutoModel, RobertaTokenizer
from sklearn.utils import resample
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "roberta-base"
TOKENIZER = RobertaTokenizer.from_pretrained(MODEL_CKPT)

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
                                    n_samples=round(len(minority_df)*1.3),    # upsample by 30%
                                    random_state=123) # reproducible results
    
    return pd.concat([majority_df, upsampled_minority_df])

def tokenize(batch):
    return TOKENIZER(text=batch["target"],
                       text_pair=batch["context"],
                       add_special_tokens=True, 
                       truncation='longest_first',
                       max_length=512, 
                       padding='max_length', 
                       return_attention_mask=True)

def yuetal_data_preprocess(gold_data_path, silver_data_path):
    """
    gold_data_path: path to gold jsonl data file
    silver_data_path: path to silver jsonl data file
    """
    gold_df = load_data_file(gold_data_path)
    silver_df = load_data_file(silver_data_path)
    silver_df = silver_df[silver_df.label == '2'] # only use minority class from silver
    
    df = pd.concat([gold_df, silver_df])
    
    # Binarize and cast label to int
    df['label'] = list(map(lambda x: 0 if int(x) == 1 or int(x) == 0 else 1, df['label']))
    
    return df

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="binary", pos_label=1)
    acc = cohen_kappa_score(labels, preds)
    roc = roc_auc_score(labels, preds)
    return {"Cohen Kappa": acc, "F1": f1, "ROC": roc}

def main():
  print("------------------- Data Loading -------------------")
  # IMPORT DATA
  if (os.path.exists(feat_path + "train_encoded.pt") and os.path.exists(feat_path + "/val_encoded.pt")):
    train_encoded = torch.load(feat_path + "train_encoded.pt")
    val_encoded = torch.load(feat_path + "val_encoded.pt")
  else:
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
    torch.save(train_encoded, feat_path + "train_encoded.pt")
    torch.save(val_encoded, feat_path + "val_encoded.pt")

    train_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

  # Import stance detection pretrained model
  model = AutoModelForSequenceClassification.from_pretrained(path + "stance_det_roberta", 
                                                          num_labels=2, ignore_mismatched_sizes=True,
                                                          hidden_dropout_prob=0.3, 
                                                          attention_probs_dropout_prob=0.25)
    
  batch_size = 16
  logging_steps = len(train_encoded) // batch_size
  model_name = MODEL_CKPT + "-finetune-2-class"    

  training_args = TrainingArguments(output_dir=model_name,
                                num_train_epochs=10,
                                learning_rate=2e-5,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=0.001,
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

  print("------------------- Testing -------------------")
  test_data = pd.read_csv(YU_DATA_PATH + "/new_test_data.csv", usecols=['context', 'target', 'Daniel annotation']).rename(columns={'Daniel annotation': 'label'})
  test_data = test_data[test_data.label.isin(['h', 'n', 'c'])]
  test_data.label = list(map(lambda x: 0 if x == 'h' or x == 'n' else 1, test_data.label))

  test_ds = Dataset.from_pandas(test_data)
  test_encoded = test_ds.map(tokenize, batched=True, batch_size=None)         

  t_preds_list = trainer.predict(test_encoded) # predict
  t_probs = t_preds_list.predictions # test probabilities
  t_preds = np.argmax(t_probs, 1) # test predictions
  t_true = t_preds_list.label_ids # true labels

  print('Test F1:', f1_score(t_true, t_preds, average="binary", pos_label=1))
  print('Test CK:', cohen_kappa_score(t_true, t_preds))
  print('Test ROC:', roc_auc_score(t_true, t_preds))    

  print("------------------- Applying BBSC -------------------")
  v_preds_list = trainer.predict(val_encoded) # validation prediction list
  v_preds = np.argmax(v_preds_list.predictions, 1) # validation predictions
  v_true = v_preds_list.label_ids # validation labels

  w = analyze_val_data(v_true, v_preds, t_preds) # generate new class weights
  new_t_preds, _ = update_probs([0,1], w, t_preds, t_probs) # generate new test predictions

  print('Test F1:', f1_score(t_true, new_t_preds, average="binary", pos_label=1))
  print('Test CK:', cohen_kappa_score(t_true, new_t_preds))
  print('Test ROC:', roc_auc_score(t_true, new_t_preds))                         

