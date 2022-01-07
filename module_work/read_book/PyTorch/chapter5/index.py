import spacy
import torchtext
import pandas as pd
import torch.nn as nn
import torch.tensor as tensor
import torch.optim as optim
from torchtext.legacy import data

# tweetsDF = pd.read_csv("./data/training.1600000.processed.noemoticon.csv",
# #                        engine="python", header=None, encoding="ISO-8859-1")
# #
# # tweetsDF["sentiment_cat"] = tweetsDF[0].astype('category')
# # tweetsDF["sentiment"] = tweetsDF["sentiment_cat"].cat.codes
# #
# # tweetsDF.to_csv("./data/train-processed.csv", header=None, index=None)
# # tweetsDF.sample(10000).to_csv("./data/train-processed-sample.csv", header=None, index=None)

LABEL = data.LabelField()
TWEET = data.Field( tokenizer_language='en_core_web_sm', lower=True)

fields = [('score', None), ('id', None), ('date', None), ('query', None),
          ('name', None), ('tweet', TWEET), ('category', None), ('label', LABEL)]

twitterDataset = data.dataset.TabularDataset(
    path="./data/train-processed.csv",
    format="CSV",
    fields=fields,
    skip_header=False)

(train, test, valid) = twitterDataset.split(split_ratio=[0.6, 0.2, 0.2],
                                            stratified=True, strata_field='label')

vocab_size = 20000
TWEET.build_vocab(train, max_size=vocab_size)
LABEL.build_vocab(train)

device = 'cuda'
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, valid, test),
    batch_size=32,
    device=device,
    sort_key=lambda x: len(x.tweet),
    sort_within_batch=False)


class OurFirstLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(OurFirstLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 2)

    def forward(self, seq):
        output, (hidden, _) = self.encoder(self.embedding(seq))
        preds = self.predictor(hidden.squeeze(0))
        return preds


model = OurFirstLSTM(100, 300, 20002)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-2)
criterion = nn.CrossEntropyLoss()


def train(epochs, model, optimizer, criterion, train_iterator, valid_iterator):
    for epoch in range(1, epochs + 1):

        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            predict = model(batch.tweet)
            loss = criterion(predict, batch.label)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * batch.tweet.size(0)
        training_loss /= len(train_iterator)

        model.eval()
        for batch_idx, batch in enumerate(valid_iterator):
            predict = model(batch.tweet)
            loss = criterion(predict, batch.label)
            valid_loss += loss.data.item() * batch.tweet.size(0)

        valid_loss /= len(valid_iterator)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}'.format(epoch, training_loss, valid_loss))

train(5, model, optimizer, criterion, train_iterator, valid_iterator)
