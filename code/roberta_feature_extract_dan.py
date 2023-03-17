from helper import yuetal_data_preprocess, extract_hidden_states, upsample, tokenize

from datasets import Dataset
import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from transformers import AutoModel, RobertaTokenizer

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YU_DATA_PATH = '../counter_context/data'
RETRAIN = False

learning_rate = 0.000001
batch_size = 64
num_epochs = 50

model_ckpt = "roberta-base"
model = AutoModel.from_pretrained("roberta-base", num_labels=2).to(DEV)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(DEV) for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def my_collate_fn(batch):
    # Process the batch data here
    inputs = []
    outputs = []

    for item in batch:
        # Extract the input and output fields from each item in the batch
        input_field = item['hidden_state']
        output_field = item['label']

        # Process the input and output fields as needed (e.g. convert to tensors)
        input_tensor = torch.tensor(input_field)
        output_tensor = torch.tensor(output_field)

        # Append the processed inputs and outputs to the lists
        inputs.append(input_tensor)
        outputs.append(output_tensor)

    # Stack the inputs and outputs into tensors and return them
    return torch.stack(inputs), torch.stack(outputs)

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    train_loss = running_loss / len(dataloader.dataset)
    train_acc = 100. * correct / total

    return train_loss, train_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            val_preds.extend(np.argmax(outputs.cpu().numpy(), axis=1).tolist())
            val_targets.extend(targets.cpu().numpy().tolist())
    
    #print(val_targets)
    #print(val_preds)
    val_roc_auc = roc_auc_score(val_targets, val_preds)
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds)
    val_roc_auc = roc_auc_score(val_targets, val_preds)


    return val_loss, val_acc, val_f1, val_roc_auc

class ClassificationHead(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 768)
        self.fc2 = nn.Linear(768, output_size)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        output = self.softmax(x)

        return output

def lr_classifier(train_hidden, val_hidden):
    X_train = np.array(train_hidden["hidden_state"])
    X_valid = np.array(val_hidden["hidden_state"])
    y_train = np.array(train_hidden["label"])
    y_valid = np.array(val_hidden["label"])
    X_train.shape, X_valid.shape

    lr_clf = LogisticRegression(max_iter=10000, verbose=1)
    lr_clf.fit(X_train, y_train)

    print("Validation Accuracy = ", lr_clf.score(X_valid, y_valid))

def main():
    if RETRAIN == False and \
        (os.path.exists("./train_hidden.pt") and os.path.exists("./val_hidden.pt")):
        train_hidden = torch.load("train_hidden.pt")
        val_hidden = torch.load("val_hidden.pt")
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
        torch.save(train_encoded, "train_encoded.pt")
        torch.save(val_encoded, "val_encoded.pt")
        
        train_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                                "label"])
        val_encoded.set_format("torch", columns=["input_ids", "attention_mask", 
                                                "label"])
        
        train_hidden = train_encoded.map(extract_hidden_states, batched=True, batch_size=32)
        val_hidden = val_encoded.map(extract_hidden_states, batched=True, batch_size=32)

        # Save the hidden state
        torch.save(train_hidden, "train_hidden.pt")
        torch.save(val_hidden, "val_hidden.pt")

    print("------------------- Training -------------------")
    train_loader = DataLoader(train_hidden, batch_size=32, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_hidden, batch_size=32, collate_fn=my_collate_fn)

    
    model = ClassificationHead(hidden_size=768, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_roc = evaluate(model, val_loader, criterion)

        print('Epoch {}/{}:'.format(epoch+1, num_epochs))
        print('Training Loss: {:.4f}\tTraining Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('Validation Loss: {:.4f}\tValidation Accuracy: {:.2f}\tValidation F1: {:.2f}\tValidation ROC: {:.2f}%'.format(val_loss, val_acc, val_f1, val_roc))
    
    '''
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for i in tqdm(range(0, len(train_hidden), batch_size)):
            # get batch of data
            inputs, labels = train_hidden[i:i+batch_size]['hidden_state'], train_hidden[i:i+batch_size]['label']
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
   '''

if __name__ == "__main__":
    main()
