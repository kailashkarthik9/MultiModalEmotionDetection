import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, BertConfig


class SentenceEmbedder:
    def __init__(self, model_name_or_path='bert-base-bert_uncased'):
        self.tokenizer, self.model = self.get_bert_model(model_name_or_path)
        self.model.eval()

    @staticmethod
    def get_bert_model(model_name_or_path):
        model_class = BertModel
        tokenizer_class = BertTokenizer
        config = BertConfig()
        config.output_hidden_states = True
        return tokenizer_class.from_pretrained(model_name_or_path), model_class.from_pretrained(model_name_or_path,
                                                                                                config=config)

    def get_encoded_tokens_from_text(self, text):
        return torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])

    def get_model_output(self, text):
        model_input = self.get_encoded_tokens_from_text(text)
        outputs = self.model(model_input)
        last_hidden_state, pooler_output, hidden_states = outputs
        hidden_states = list(hidden_states)[1:]
        return hidden_states, pooler_output

    @staticmethod
    def plot_hidden_state_vector(hidden_states, token_i, layer_i):
        vec = hidden_states[layer_i][0][token_i]
        # Plot the values as a histogram to show their distribution.
        plt.figure(figsize=(10, 10))
        plt.hist(vec, bins=200)
        plt.show()

    @staticmethod
    def get_token_major_hidden_states(hidden_states):
        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings_re_ordered = token_embeddings.permute(1, 0, 2)
        return token_embeddings_re_ordered

    def get_word_vectors(self, text):
        hidden_states, _ = self.get_model_output(text)
        token_major_hidden_states = self.get_token_major_hidden_states(hidden_states)
        # Sum of last four layers
        token_vectors_sum = []
        for token in token_major_hidden_states:
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vectors_sum.append(sum_vec)
        return token_vectors_sum

    def get_sentence_vector(self, text):
        hidden_states, _ = self.get_model_output(text)
        token_vectors = hidden_states[11][0]
        return torch.mean(token_vectors, dim=0)

    def embed_dataset(self, directory):
        file_paths = glob.glob(directory + '/*')
        for file_path in file_paths:
            if os.path.isdir(file_path):
                continue
            if not file_path.endswith('.csv'):
                continue
            data = pd.read_csv(file_path)
            data['TextEmbeddings'] = data['Utterance'].apply(lambda x: self.get_sentence_vector(x).tolist())
            data.to_csv(file_path[:-4] + '_word_embedded.csv', index=False)


if __name__ == '__main__':
    embedder = SentenceEmbedder('bert/uncased/')
    sentence_embedding = embedder.get_sentence_vector('Here is the sentence I want embeddings for to give to Kailash.')
    print(sentence_embedding.tolist())
    embedder.embed_dataset('iemocap/data')
    embedder.embed_dataset('meld/data')
