from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.datasets import fetch_20newsgroups
import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
model.to(device)

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)

print(twenty_train.data[0])  # pour afficher le premier texte
len(twenty_train.target)  # pour afficher le premier label

data_couche_n = []

batches = []
batch_size = 32

nb_batch = 64
print("Tokenizing...")
for i in tqdm(range(min(len(twenty_train.data) // batch_size, nb_batch))):

    # print(len(twenty_train.data) // batch_size, i)
    maxlength = 0
    batch = []
    for j in range(batch_size):
        batch.append(tokenizer(twenty_train.data[i * batch_size + j], return_tensors="pt").input_ids[0])
        maxlength = max(maxlength, len(batch[-1]))

    batch_tensor = torch.zeros((batch_size, 512))
    for j in range(batch_size):
        if len(batch[j]) > 512:
            sentence_ids = batch[j][:512]
        else:
            sentence_ids = batch[j]
        batch_tensor[j, :len(sentence_ids)] = sentence_ids.to(device)

    batches.append(batch_tensor.type(torch.int64))
# batches = torch.stack(batches).to(device)
print('\nevaluating...')
for batch in tqdm(batches[:nb_batch]):
    batch = batch.to(device)
    model.eval()
    with torch.no_grad():
        model_output = model(batch, output_hidden_states=True)
        hidden_states = model_output.hidden_states
        for i, partial_output in enumerate(hidden_states):
            last_token = (partial_output.cpu())[:, -1]
            try:
                data_couche_n[i].append(torch.reshape(last_token, (-1, 768)))
            except:
                data_couche_n.append([torch.reshape(last_token, (-1, 768))])
        batch.cpu()
        print(len(data_couche_n) * len(data_couche_n))

x_n = [torch.zeros((nb_batch * batch_size, 768)) for i in range(13)]
print('\nreshaping...')
for i in tqdm(range(13)):
    for j in range(nb_batch):
        x_n[i][j * batch_size: (j + 1) * batch_size] = data_couche_n[i][j]

y = twenty_train.target[:nb_batch * batch_size]
print(y)


# Entrée : 1 hidden_state : nb token max * embeding_size
# Sortie : 4 classes

# X entrée, y sortie
def test_classifier(x_train, y_train, x_val, y_val):
    classifier = LogisticRegression(verbose=False).fit(x_train, y_train)
    prediction = classifier.predict(x_val)
    score = (prediction == y_val).sum() / len(y_val)
    return score


def test_layers(layers_Xs, target_y):
    results = []
    y_train, y_val = train_test_split(target_y, shuffle=False)
    print("testing...")
    for X in tqdm(layers_Xs):
        x_train, x_val = train_test_split(X, shuffle=False)
        results.append(test_classifier(x_train, y_train, x_val, y_val))
    plt.plot(results)
    plt.show()
    return results


res = test_layers(x_n, y)
