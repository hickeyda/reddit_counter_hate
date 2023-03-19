import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.000001
batch_size = 64
num_epochs = 50

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

def main():
    print("------------------- Data Loading -------------------")
    train_hidden = torch.load("train_hidden.pt")
    val_hidden = torch.load("val_hidden.pt")
    
    train_loader = DataLoader(train_hidden, batch_size=32, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_hidden, batch_size=32, collate_fn=my_collate_fn)

    print("------------------- Training -------------------")
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
