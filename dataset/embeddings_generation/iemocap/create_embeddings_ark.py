import os
from ast import literal_eval

import kaldi_io
import numpy as np
import pandas as pd

from classifier.classifier_lda_plda import EmbeddingSource

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


class SessionArkReactor:
    def __init__(self, ark_files, embeddings_file, embedding_source: EmbeddingSource):
        self.embedding_source = embedding_source
        self.speech_arks = [(os.path.basename(file_), kaldi_io.read_vec_flt_ark(file_)) for file_ in ark_files]
        self.multi_modal_embeddings = pd.read_csv(embeddings_file)
        self.multi_modal_embeddings[embedding_source.value] = self.multi_modal_embeddings[embedding_source.value].apply(
            literal_eval)

    @staticmethod
    def cluster_arks(arks):
        combined_ark = {
            '1': list(),
            '2': list(),
            '3': list(),
            '4': list(),
            '5': list(),
        }
        for ark_file, ark in arks:
            for key, vector in ark:
                combined_ark[SessionArkReactor.get_session_number(key)].append((key, vector))
        return combined_ark

    @staticmethod
    def get_session_number(ark_key):
        # anger/disgust-test-01F-improvisation-01-000-M
        return SessionArkReactor.decompose_ark_key(ark_key)[0]

    @staticmethod
    def decompose_ark_key(key):
        # anger/disgust-test-01F-improvisation-01-000-M
        key_components = key.split('-')
        return key_components[2][1], key_components[2][2], key_components[3], key_components[4], key_components[5], \
               key_components[6]

    def get_modified_file_name(self, session):
        return self.embedding_source.value + '_xvector.' + session + '.ark'

    def get_modified_ark_components(self, speech_arks):
        modified_arks = []
        for session, speech_ark in speech_arks.items():
            ark = []
            for key, array_ in speech_ark:
                session_number, mocap_source, dialogue_type, dialogue_number, utterance_number, speaker = \
                    self.decompose_ark_key(key)
                embeddings_record = self.multi_modal_embeddings[
                    (self.multi_modal_embeddings['Session_Number'] == int(session_number)) & (
                            self.multi_modal_embeddings['Mocap_Source'] == mocap_source) & (
                            self.multi_modal_embeddings['Dialogue_Type'] == dialogue_type) & (
                            self.multi_modal_embeddings['Dialogue_Number'] == dialogue_number) & (
                            self.multi_modal_embeddings['Utterance_Number'] == int(utterance_number)) & (
                            self.multi_modal_embeddings['Speaker'] == speaker)]
                embedding = embeddings_record[self.embedding_source.value].values[0]
                ark.append((key, embedding))
            modified_arks.append((session, ark))
        return modified_arks

    def create_session_wise_arks(self):
        session_wise_speech_arks = self.cluster_arks(self.speech_arks)
        modified_ark_components = self.get_modified_ark_components(session_wise_speech_arks)
        for session, components in modified_ark_components:
            with open('iemocap/data/kaldi/modified/' + self.get_modified_file_name(session), 'wb') as file_:
                for key, vector in components:
                    kaldi_io.write_vec_flt(file_, np.array(vector), key)


if __name__ == '__main__':
    ark_files_ = [
        'iemocap/data/kaldi/xvector.1.ark',
        'iemocap/data/kaldi/xvector.2.ark',
        'iemocap/data/kaldi/xvector.3.ark',
        'iemocap/data/kaldi/xvector.4.ark',
        'iemocap/data/kaldi/xvector.5.ark',
    ]
    embeddings_file_ = 'iemocap/data/dataset_with_multi_modal_embeddings.csv'
    reactor = SessionArkReactor(ark_files_, embeddings_file_, EmbeddingSource.TEXT)
    reactor.create_session_wise_arks()
    reactor = SessionArkReactor(ark_files_, embeddings_file_, EmbeddingSource.SPEECH)
    reactor.create_session_wise_arks()
    reactor = SessionArkReactor(ark_files_, embeddings_file_, EmbeddingSource.MULTIMODAL)
    reactor.create_session_wise_arks()
