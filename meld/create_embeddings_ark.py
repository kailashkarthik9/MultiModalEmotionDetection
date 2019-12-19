import os
from ast import literal_eval

import kaldi_io
import numpy as np
import pandas as pd

from classifier_lda_plda import EmbeddingSource

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


class ArkReactor:
    def __init__(self, ark_files, embeddings_file, embedding_source: EmbeddingSource):
        self.embedding_source = embedding_source
        self.speech_arks = [(os.path.basename(file_), kaldi_io.read_vec_flt_ark(file_)) for file_ in ark_files]
        self.multi_modal_embeddings = pd.read_csv(embeddings_file)
        self.multi_modal_embeddings[embedding_source.value] = self.multi_modal_embeddings[embedding_source.value].apply(
            literal_eval)

    def get_modified_ark_components(self):
        modified_arks = []
        for file_name, speech_ark in self.speech_arks:
            ark = []
            for key, array_ in speech_ark:
                dataset_split, dialogue_id, utterance_id = self.decompose_ark_key(key)
                embeddings_record = self.multi_modal_embeddings[
                    (self.multi_modal_embeddings['Dataset_Split'] == dataset_split) & (
                            self.multi_modal_embeddings['Dialogue_ID'] == int(dialogue_id)) & (
                            self.multi_modal_embeddings['Utterance_ID'] == int(utterance_id))]
                embedding = embeddings_record[self.embedding_source.value].values[0]
                ark.append((key, embedding))
            modified_arks.append((file_name, ark))
        return modified_arks

    def create_modified_ark(self):
        modified_ark_components = self.get_modified_ark_components()
        for file_name, components in modified_ark_components:
            with open('data/kaldi/modified/' + self.get_modified_file_name(file_name), 'wb') as file_:
                for key, vector in components:
                    kaldi_io.write_vec_flt(file_, np.array(vector), key)

    @staticmethod
    def decompose_ark_key(key):
        # fear/surprise-dev-0-1
        key_components = key.split('-')
        return key_components[1], key_components[2], key_components[3]

    def get_modified_file_name(self, file_name):
        return self.embedding_source.value + '_' + file_name


if __name__ == '__main__':
    ark_files_ = [
        'data/kaldi/xvector.1.ark',
        'data/kaldi/xvector.2.ark',
        'data/kaldi/xvector.3.ark',
        'data/kaldi/xvector.4.ark',
        'data/kaldi/xvector.5.ark',
    ]
    embeddings_file_ = 'data/dataset_with_multi_modal_embeddings.csv'
    reactor = ArkReactor(ark_files_, embeddings_file_, EmbeddingSource.SPEECH)
    reactor.create_modified_ark()
    reactor = ArkReactor(ark_files_, embeddings_file_, EmbeddingSource.TEXT)
    reactor.create_modified_ark()
    reactor = ArkReactor(ark_files_, embeddings_file_, EmbeddingSource.MULTIMODAL)
    reactor.create_modified_ark()
