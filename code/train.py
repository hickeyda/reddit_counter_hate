import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import json
from datasets import load_dataset, DatasetDict, Dataset

from transformers import RobertaTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = './data'

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
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

batch_size = 64
logging_steps = len(train_encoded) // batch_size
print(logging_steps)
model_name = f"{model_ckpt}-finetuned-counter-hate"
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
                                  log_level="error")

trainer = Trainer(model=model, args=training_args, 
                  compute_metrics=compute_metrics,
                  train_dataset=train_encoded,
                  eval_dataset=val_encoded,
                  tokenizer=tokenizer)
trainer.train()
trainer.save_model("model")
