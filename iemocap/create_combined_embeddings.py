from ast import literal_eval

import pandas as pd

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


class MultiModalEmbeddingGeneratorIemocap:
    def __init__(self, speech_vectors_file, text_vectors_files):
        self.speech_embeddings = self.get_speech_vectors(speech_vectors_file)
        self.text_embeddings = self.get_text_vectors(text_vectors_files)

    @staticmethod
    def get_speech_vectors(speech_vectors_file):
        df = pd.read_csv(speech_vectors_file)
        df['xvector'] = df['xvector'].apply(literal_eval)
        df['Session_Number'] = df['session_id_mocap'].apply(lambda x: x[1])
        df['Mocap_Source'] = df['session_id_mocap'].apply(lambda x: x[2])
        df['utterance_id'] = df['utterance_id'].apply(str)
        df = df[['Session_Number', 'Mocap_Source', 'dialogue_type', 'dialogue_id', 'utterance_id', 'gender', 'xvector']]
        df.columns = ['Session_Number', 'Mocap_Source', 'Dialogue_Type', 'Dialogue_Number', 'Utterance_Number',
                      'Speaker', 'SpeechEmbeddings']
        return df

    @staticmethod
    def get_text_vectors(text_vectors_files):
        session_1_df = pd.read_csv(text_vectors_files['Session1'])
        session_2_df = pd.read_csv(text_vectors_files['Session2'])
        session_3_df = pd.read_csv(text_vectors_files['Session3'])
        session_4_df = pd.read_csv(text_vectors_files['Session4'])
        session_5_df = pd.read_csv(text_vectors_files['Session5'])
        session_1_df['TextEmbeddings'] = session_1_df['TextEmbeddings'].apply(literal_eval)
        session_2_df['TextEmbeddings'] = session_2_df['TextEmbeddings'].apply(literal_eval)
        session_3_df['TextEmbeddings'] = session_3_df['TextEmbeddings'].apply(literal_eval)
        session_4_df['TextEmbeddings'] = session_4_df['TextEmbeddings'].apply(literal_eval)
        session_5_df['TextEmbeddings'] = session_5_df['TextEmbeddings'].apply(literal_eval)
        session_1_df['Session_Number'] = session_1_df['Session_Number'].apply(str)
        session_2_df['Session_Number'] = session_2_df['Session_Number'].apply(str)
        session_3_df['Session_Number'] = session_3_df['Session_Number'].apply(str)
        session_4_df['Session_Number'] = session_4_df['Session_Number'].apply(str)
        session_5_df['Session_Number'] = session_5_df['Session_Number'].apply(str)
        session_1_df['Utterance_Number'] = session_1_df['Utterance_Number'].apply(str)
        session_2_df['Utterance_Number'] = session_2_df['Utterance_Number'].apply(str)
        session_3_df['Utterance_Number'] = session_3_df['Utterance_Number'].apply(str)
        session_4_df['Utterance_Number'] = session_4_df['Utterance_Number'].apply(str)
        session_5_df['Utterance_Number'] = session_5_df['Utterance_Number'].apply(str)
        return pd.concat([session_1_df, session_2_df, session_3_df, session_4_df, session_5_df], ignore_index=True)

    def get_multimodal_embeddings(self):
        combined_df = pd.merge(self.text_embeddings, self.speech_embeddings,
                               on=['Session_Number', 'Mocap_Source', 'Dialogue_Type', 'Dialogue_Number',
                                   'Utterance_Number', 'Speaker'])
        assert len(combined_df) != len(self.text_embeddings)
        assert len(combined_df) == len(self.speech_embeddings)
        combined_df['MultimodalEmbeddings'] = combined_df[['TextEmbeddings', 'SpeechEmbeddings']].apply(
            lambda x: x[0] + x[1], axis=1)
        return combined_df  # .drop(['TextEmbeddings', 'SpeechEmbeddings'], axis=1)

    def create_multimodal_embeddings(self, output_file):
        df = self.get_multimodal_embeddings()
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    speech_vectors_file_ = 'data/iemocap_xvectors.csv'
    text_vectors_files_ = {
        'Session1': 'data/Session1_word_embedded.csv',
        'Session2': 'data/Session2_word_embedded.csv',
        'Session3': 'data/Session3_word_embedded.csv',
        'Session4': 'data/Session4_word_embedded.csv',
        'Session5': 'data/Session5_word_embedded.csv',
    }
    generator = MultiModalEmbeddingGeneratorIemocap(speech_vectors_file_, text_vectors_files_)
    # pickle.dump(generator, open('generator.pkl', 'wb'))
    # generator = pickle.load(open('generator.pkl', 'rb'))
    generator.create_multimodal_embeddings('data/dataset_with_multi_modal_embeddings.csv')
    print('ok')
