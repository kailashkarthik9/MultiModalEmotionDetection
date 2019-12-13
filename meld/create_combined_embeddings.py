from ast import literal_eval

import pandas as pd


class MultiModalEmbeddingGenerator:
    def __init__(self, speech_vectors_file, text_vectors_files):
        self.speech_embeddings = self.get_speech_vectors(speech_vectors_file)
        self.text_embeddings = self.get_text_vectors(text_vectors_files)

    def get_speech_vectors(self, speech_vectors_file):
        df = pd.read_csv(speech_vectors_file)
        df['xvector'] = df['xvector'].apply(literal_eval)
        df = df[['set', 'dialogue_id', 'utterance_id', 'label', 'xvector']]
        df.columns = ['Dataset_Split', 'Dialogue_ID', 'Utterance_ID', 'Emotion_Label', 'SpeechEmbeddings']
        return df

    def get_text_vectors(self, text_vectors_files):
        dev_df = pd.read_csv(text_vectors_files['dev'])
        train_df = pd.read_csv(text_vectors_files['train'])
        test_df = pd.read_csv(text_vectors_files['test'])
        dev_df.loc[:, 'Dataset_Split'] = 'dev'
        train_df.loc[:, 'Dataset_Split'] = 'train'
        test_df.loc[:, 'Dataset_Split'] = 'test'
        dev_df['TextEmbeddings'] = dev_df['TextEmbeddings'].apply(literal_eval)
        train_df['TextEmbeddings'] = train_df['TextEmbeddings'].apply(literal_eval)
        test_df['TextEmbeddings'] = test_df['TextEmbeddings'].apply(literal_eval)
        return pd.concat([dev_df, train_df, test_df], ignore_index=True)

    def get_multimodal_embeddings(self):
        combined_df = pd.merge(self.text_embeddings, self.speech_embeddings,
                               on=['Dataset_Split', 'Dialogue_ID', 'Utterance_ID'])
        assert len(combined_df) == len(self.text_embeddings)
        assert len(combined_df) == len(self.speech_embeddings)
        combined_df['Multimodal_Embeddings'] = combined_df[['TextEmbeddings', 'SpeechEmbeddings']].apply(
            lambda x: x[0].extend(x[1]), axis=1)
        return combined_df.drop(['TextEmbeddings', 'SpeechEmbeddings'])

    def create_multimodal_embeddings(self, output_file):
        df = self.get_multimodal_embeddings()
        df.to_csv(output_file)


if __name__ == '__main__':
    speech_vectors_file_ = 'data/meld_xvectors.csv'
    text_vectors_files_ = {
        'dev': 'data/dev_sent_emo_norm_word_embedded.csv',
        'train': 'data/train_sent_emo_norm_word_embedded.csv',
        'test': 'data/test_sent_emo_norm_word_embedded.csv',
    }
    generator = MultiModalEmbeddingGenerator(speech_vectors_file_, text_vectors_files_)
    generator.create_multimodal_embeddings('data/dataset_with_multi_modal_embeddings.csv')
    print('ok')
