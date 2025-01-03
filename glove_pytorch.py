import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
import pandas as pd
import re
from collections import Counter
from torchtext.vocab import vocab
import numpy as np
from sklearn import metrics

class LSTM_Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,pretrained_embeddings=None):
        super(LSTM_Net, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embed.weight.requires_grad=False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Za NLLLoss()

    def forward(self, x):
        z = self.embed(x)
        lstm_out, (hidden, _) = self.lstm(z)  
        z = hidden[-1]  # Uzima poslednji hidden state: umesto torch.Size([1,32, 100]) sa ovim je torch.Size([32, 100])
        #print(hidden.shape,hidden[:,:],"\nlstm:",lstm_out.shape,lstm_out[:,-1,:],"\nZ: ",z.shape,z,
        #      lstm_out[-1].shape,lstm_out[-1],"\n\n====================================\n")
        z = self.log_softmax(self.fc1(z))
        return z

class DatasetImdb:
    def __init__(self, csv_path, max_vocab_size=6000):
        self.data = pd.read_csv(csv_path)
        self.data["sentiment"] = self.data["sentiment"].apply(lambda i: 1 if i == "positive" else 0)
        self.x = self.data["review"].apply(self._clean_text).tolist()
        self.y = torch.tensor(self.data["sentiment"].tolist())
        self.vocab = self._build_vocab(self.x, max_vocab_size)
        self.tokenizer = get_tokenizer("basic_english")

    def _clean_text(self, text):
        return re.sub(r'<.*?>', '', text)

    def _build_vocab(self, data, max_vocab_size):
        counter = Counter()
        for text in data:
            tokens = text.lower().split()
            counter.update(tokens)
        most_common = counter.most_common(max_vocab_size)
        vocab_obj = vocab(dict(most_common), specials=["<pad>", "<unk>"])
        vocab_obj.set_default_index(vocab_obj["<unk>"])
        return vocab_obj

    def tokenized_and_transformed(self, data):
        tokenized = [self.tokenizer(text) for text in data]
        transformed = [[self.vocab[token] for token in tokens] for tokens in tokenized]
        return transformed

# ucitavanje i tokenizacija
csv_path = "C:/Users/Djole/Desktop/IMDB Dataset.csv"
dataset = DatasetImdb(csv_path)
x_transformed = dataset.tokenized_and_transformed(dataset.x)

# padding
max_len = 500
x_tensor = [torch.tensor(seq[:max_len]) for seq in x_transformed]
x_padded = pad_sequence(x_tensor, batch_first=True, padding_value=dataset.vocab["<pad>"])
train_dataset = TensorDataset(x_padded, dataset.y)

# split skupa na train i valid skup
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

# dataloader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)


glove={}
with open("C:/Users/Djole/Desktop/glove.6B.100d.txt",encoding="utf-8") as f:
    for linija in f:
        red=linija.split()
        rec=red[0]
        vektor=red[1:]
        #print(red,"REC: ",rec,"\n\nVektor",vektor)
        vektor=np.asarray(vektor,dtype="float32")
        glove[rec]=vektor

#================================================================================================
moj_glove=np.zeros((6002,100))  # ((len(dataset.vocab),100))

moj_glove[0]=np.zeros(100)
moj_glove[1]=np.random.uniform(-0.5,0.5,100)

for rec,idx in dataset.vocab.get_stoi().items():
    vektor_glove=glove.get(rec)
    if vektor_glove is not None:
        moj_glove[idx]=vektor_glove
    else:
        moj_glove[idx]= moj_glove[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Net(vocab_size=len(dataset.vocab), embedding_dim=100, hidden_dim=100, output_dim=2,pretrained_embeddings=moj_glove).to(device)
loss_func = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=7e-4)

from torch.optim.lr_scheduler import StepLR
scheduler= StepLR(optimizer, step_size=20,gamma=0.9)

max_epochs = 200
model.train()
for epoch in range(max_epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_func(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    sch=optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}, sch: {sch:.10f}")


with torch.no_grad():
    y_true = []
    y_predicted = []
    
    for batch in val_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        predictions = torch.argmax(outputs, dim=1)  
        
        y_true.extend(y.cpu().numpy())  
        y_predicted.extend(predictions.cpu().numpy())  
    
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)

    rez = metrics.confusion_matrix(y_true, y_predicted)
    print("Matrica konfuzije: ")
    print(rez)
