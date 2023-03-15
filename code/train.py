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

from preprocess_data import yuetal_data_preprocess

def load_data_file(path):
    lines_list = []
    with open(path, 'r') as f:
        for line in f:
            lines_list.append(json.loads(line))

    df = pd.DataFrame(lines_list)
    
    return df

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, add_special_tokens = True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YU_DATA_PATH = '../reference/counter_context/data'
RETRAIN = True

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
        
    else:
        pass
    
    model_ckpt = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", 
                                                            num_labels=2)

    batch_size = 4
    logging_steps = len(train_encoded) // batch_size
    print(logging_steps)
    model_name = model_ckpt + "-finetune"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=8,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 2)

    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=train_encoded,
                    eval_dataset=val_encoded,
                    tokenizer=tokenizer)
    
    print("------------------- Training -------------------")
    trainer.train()
    trainer.save_model("roberta-fine-tune-model")

if __name__ == "__main__":
    main()
    