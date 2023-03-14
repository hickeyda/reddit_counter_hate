import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import json
from datasets import load_dataset, DatasetDict, Dataset

from transformers import RobertaTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModel
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '../reference/counter_context/data'

def load_data_file(path):
    lines_list = []
    with open(path, 'r') as f:
        for line in f:
            lines_list.append(json.loads(line))

    df = pd.DataFrame(lines_list)
    
    return df

def preprocess(df):
    # Combine context and target
    contexts = df['context'].values
    targets = df['target'].values
    df['text'] = list(map(lambda x,y: str(x) + ' [SEP] ' + str(y), contexts, targets))
    df = df.drop(columns=['context', 'target'])
    
    # Cast label to int
    df['label'] = list(map(lambda x: int(x), df['label']))
    
    # TODO: Binarize labels
    
    return df

train_gold_df = load_data_file(data_path + '/gold/train.jsonl')
val_gold_df = load_data_file(data_path + '/gold/val.jsonl')
test_df = load_data_file(data_path + '/gold/test.jsonl') # The test set only comes from the 'gold' category

train_silver_df = load_data_file(data_path + '/silver/train.jsonl')
val_silver_df = load_data_file(data_path + '/silver/val.jsonl')

#Combine Gold and Silver
train_df = pd.concat([train_gold_df, train_silver_df])
val_df = pd.concat([val_gold_df, val_silver_df])

train_df = preprocess(train_df)
val_df = preprocess(val_df)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, add_special_tokens = True)

train_encoded = train_ds.map(tokenize, batched=True, batch_size=None)
val_encoded = train_ds.map(tokenize, batched=True, batch_size=None)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

model_ckpt = "roberta-base"
model = AutoModel.from_pretrained("roberta-base", num_labels=3).to(dev)

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(dev) for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

train_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                           "label"])

train_hidden = train_encoded.map(extract_hidden_states, batched=True)
val_hidden = val_encoded.map(extract_hidden_states, batched=True)


X_train = np.array(train_hidden["hidden_state"])
X_valid = np.array(val_hidden["hidden_state"])
y_train = np.array(train_hidden["label"])
y_valid = np.array(val_hidden["label"])
X_train.shape, X_valid.shape

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)

lr_clf.score(X_valid, y_valid)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib as plt

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
    
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, train_hidden.features["label"].names)
