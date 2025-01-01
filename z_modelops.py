import json
import numpy as np
from torch import nn 
import torch
from torch.utils.data import random_split, DataLoader
from z_dataops import NamesDataset, transform, proxy_collate_batch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import string

class NameToLanguages(nn.Module):
    def __init__(self, feature_size=26, n_classes=18):
        super().__init__()

        # create simple architecture 
        self.net_rnn = nn.RNN(input_size=feature_size, hidden_size=128, batch_first=True)
        self.net_linear = nn.Linear(in_features=128, out_features=n_classes)
        
    def forward(self, x): 
        rnn_out, last_ts = self.net_rnn(x)
        output = self.net_linear(last_ts[0])
        return output
    
def training(model: nn.Module, train_batch: list, optimizer, loss_fn): 
    model.train() 
    batch_loss = 0

    for x, y in train_batch: 
        # predict
        y_pred = model(x) 
        # compute loss
        curr_loss = loss_fn(y_pred, y)
        batch_loss += curr_loss

    # reset grad
    optimizer.zero_grad()
    # calculate grad
    batch_loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), 3)
    # step 
    optimizer.step()

    return batch_loss.item() / len(train_batch)

def validation(model, dl: DataLoader, loss_fn): 
    model.eval() 
    batch_loss = 0
    with torch.no_grad():
        for item in dl: 
            for x, y in item: 
                # predict 
                y_pred = model(x) 
                # loss
                curr_loss = loss_fn(y_pred, y)
                batch_loss += curr_loss
    return batch_loss.item() / len(dl)

def plot_losses(loss_label, title, save_location="model/loss.png"): 
    for k, v in loss_label.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.title(title)
    plt.savefig(save_location)

def load_labels(input_file="model/label.json"):
    # Read the dictionary from the file
    with open(input_file, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def evaluate(rnn, validation_dl, classes):
    # CODE AS IS FROM: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#evaluating-the-results
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for item in validation_dl:
            for text_tensor, label in item:
                output = rnn(text_tensor)
                #
                _, idx = output.topk(1) 
                guess, guess_i = classes[str(idx.item())], idx.item()
                label_i = label.item()
                confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy()) #numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    tag = [classes[str(i)] for i in range(len(classes))]
    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=tag, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=tag)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig("model/evaluate.png")

def load_labels(input_file="model/label.json"):
    # Read the dictionary from the file
    with open(input_file, 'r') as file:
        dictionary = json.load(file)
    return dictionary

if __name__=="__main__": 
    model = NameToLanguages(feature_size=len(string.ascii_letters))

    # #Sanity Check Model
    # x = torch.randn((1, 7, 26)) # (batch, word_length, one-hot-ascii-char)
    # model.eval()
    # with torch.no_grad():
    #     out = model(x)
    #     print(out.shape)

    # #Optimziers, Loss 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n_epoch = 27

    # #Training Loop 
    ds = NamesDataset(transform=transform)
    train_ds, val_ds = random_split(ds, [0.7, 0.3], generator=torch.Generator().manual_seed(31))
    train_dl    = DataLoader(dataset=train_ds, batch_size=64, collate_fn=proxy_collate_batch)
    val_dl      = DataLoader(dataset=val_ds, collate_fn=proxy_collate_batch)
    # #Trackers 
    train_losses, val_losses = [], []

    for epoch in range(n_epoch):
        for batch in train_dl: 
            train_loss = training(model, batch, optimizer, loss_fn)
            # report val loss 
            
        train_losses.append(train_loss)
        val_loss = validation(model, val_dl, loss_fn)
        val_losses.append(val_loss)


        print(f"Epoch {epoch}: Train_loss: {train_losses[-1]}, Val_loss: {val_loss}")
    plot_losses({"train": train_losses, "val": val_losses}, "Training Loss")
    torch.save(model, "model/rnn.pth")

    classes = load_labels()
    evaluate(model, val_dl, classes)

