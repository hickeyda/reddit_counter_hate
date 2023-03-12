import pandas as pd
from transformers import RobertaTokenizer
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import json

'''
Code adapted from the Github repository for "Pinpointing Fine-Grained Relationships between Hateful Tweets and Replies"
AAAI 2022
https://github.com/albanyan/hateful-tweets-replies/blob/main/Code/prepare_data.py
'''

DATA_PATH = 'PATH_TO_COUNTER_CONTEXT_DATA'
OUTPUT_PATH = 'DIR_TO_SAVE_DATA'

# Constant seed for reproducibility.
SEED = 42
BATCH_SIZE = 8

def roberta_encode(df, tokenizer, max_seq_length=512):
    input_ids = []
    attention_masks = []
    for sent in df[['context', 'target']].values:
        sent = sent[0] + ' [SEP] ' +  sent[1]
        encoded_dict = tokenizer.encode_plus(
			sent,                      # Sentence to encode.
			add_special_tokens = True, # Add '[CLS]' and '[SEP]'
			max_length = max_seq_length,           # Pad & truncate all sentences.
			pad_to_max_length = True,
			return_attention_mask = True,   # Construct attn. masks.
			return_tensors = 'pt',     # Return pytorch tensors.
		)
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
    'input_word_ids': input_ids,
    'input_mask': attention_masks}

    return inputs

def load_data_file(path):
    lines_list = []
    with open(path, 'r') as f:
        for line in f:
            lines_list.append(json.loads(line))

    df = pd.DataFrame(lines_list)
    
    return df

def binarize_label(label):
    if int(label) == 1:      #"Neutral" label combines with hate to form "not counter-hate"
        label = 0

    return int(label)

def main():
    data_path = DATA_PATH

    #"Gold" and "Silver" directories refer to annotated data with different levels of inter-annotator agreement
    train_gold_df = load_data_file(data_path + '/gold/train.jsonl')
    val_gold_df = load_data_file(data_path + '/gold/val.jsonl')
    test_df = load_data_file(data_path + '/gold/test.jsonl') # The test set only comes from the 'gold' category

    train_silver_df = load_data_file(data_path + '/silver/train.jsonl')
    val_silver_df = load_data_file(data_path + '/silver/val.jsonl')

    #Combine Gold and Silver
    train_df = pd.concat([train_gold_df, train_silver_df])
    val_df = pd.concat([val_gold_df, val_silver_df])


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train = roberta_encode(train_df, tokenizer)
    train_labels = train_df['label'].apply(binarize_label)

    val = roberta_encode(val_df, tokenizer)
    val_labels = val_df['label'].apply(binarize_label)

    test = roberta_encode(test_df, tokenizer)
    test_labels = test_df['label'].apply(binarize_label)
    

    #Combine the inputs into a TensorDataset
    input_ids, attention_masks = train.values()
    labels = torch.tensor(train_labels.values,dtype=torch.long)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)

    input_ids, attention_masks = val.values()
    labels = torch.tensor(val_labels.values,dtype=torch.long)
    val_dataset = TensorDataset(input_ids, attention_masks, labels)

    input_ids, attention_masks = test.values()
    labels = torch.tensor(test_labels.values,dtype=torch.long)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    
    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = BATCH_SIZE # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = BATCH_SIZE # Evaluate with this batch size.
            )

    # For testing the order doesn't matter, so we'll just read them sequentially.
    testing_dataloader = DataLoader(
                test_dataset, # The test samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = BATCH_SIZE # Evaluate with this batch size.
            )
    
    output_dir = OUTPUT_PATH

    torch.save(train_dataloader, os.path.join(output_dir, 'train.pth'))
    torch.save(validation_dataloader, os.path.join(output_dir, 'valid.pth'))
    torch.save(testing_dataloader, os.path.join(output_dir, 'test.pth'))

if __name__ == "__main__":
  main()