import pickle

import pandas as pd

conversation_length = pickle.load(open('valid/conversation_length.pkl', 'rb'))
labels = pickle.load(open('valid/labels.pkl', 'rb'))
sentences = pickle.load(open('valid/sentences.pkl', 'rb'))
sentence_length = pickle.load(open('valid/sentence_length.pkl', 'rb'))

data = list()
for sentences_, labels_, sentence_length_ in zip(sentences, labels, sentence_length):
    for sentence, label, length in zip(sentences_, labels_, sentence_length_):
        sentence_text = ' '.join(sentence[:length - 1])
        data.append({
            'sentence': sentence_text,
            'label': label
        })
df_data = pd.DataFrame(data)
df_data.to_csv('valid.csv')
print('dataset explored!')
