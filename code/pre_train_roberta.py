from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

'''
Code largely adapted from the fine-tuning tutorial on huggingface: https://huggingface.co/docs/transformers/training
'''

# stance_train.csv and stance_val.csv are a 90-10 random split from the DEBAGREEMENT dataset
# The files are too big for github. I also added a field 'all_text' to the csv which is parent + ' [SEP] ' + child
dataset = load_dataset("csv", data_files={'train': './stance_train.csv', 'validation': './stance_val.csv'})

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples["all_text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

training_args = TrainingArguments(output_dir="stance_training")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

# Each time a checkpoint is saved, it takes up 1.7GB of disk space. Max storage on OSU HPC is 15GB.
# save_total_limit automatically deletes these checkpoints if the number of them exceeds the limit.
training_args = TrainingArguments(output_dir="stance_trainer", evaluation_strategy="epoch", save_steps=1500,
                                  save_total_limit=3, resume_from_checkpoint=True)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_datasets['train'],

    eval_dataset=tokenized_datasets['validation'],

    compute_metrics=compute_metrics,

)

trainer.train()

trainer.save_model('./fine_tuned_for_stance')