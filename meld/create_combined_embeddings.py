from ast import literal_eval

import pandas as pd


class MultiModalEmbeddingGeneratorMeld:
    def __init__(self, speech_vectors_file, text_vectors_files):
        self.speech_embeddings = self.get_speech_vectors(speech_vectors_file)
        self.text_embeddings = self.get_text_vectors(text_vectors_files)

    @staticmethod
    def get_speech_vectors(speech_vectors_file):
        df = pd.read_csv(speech_vectors_file)
        df['xvector'] = df['xvector'].apply(literal_eval)
        df = df[['set', 'dialogue_id', 'utterance_id', 'label', 'xvector']]
        df.columns = ['Dataset_Split', 'Dialogue_ID', 'Utterance_ID', 'Emotion_Label', 'SpeechEmbeddings']
        return df

    @staticmethod
    def get_text_vectors(text_vectors_files):
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
        assert len(combined_df) != len(self.text_embeddings)
        assert len(combined_df) == len(self.speech_embeddings)
        combined_df['MultimodalEmbeddings'] = combined_df[['TextEmbeddings', 'SpeechEmbeddings']].apply(
            lambda x: x[0] + x[1], axis=1)
        combined_df.columns = ['Sr No.', 'Utterance', 'Speaker', 'Emotion', 'Sentiment', 'Dialogue_ID', 'Utterance_ID',
                               'Season', 'Episode', 'StartTime', 'EndTime', 'Emotion_Label', 'TextEmbeddings',
                               'Dataset_Split', 'Emotion_Label_y', 'SpeechEmbeddings', 'MultimodalEmbeddings']
        combined_df = combined_df.drop(['Emotion_Label_y'], axis=1)
        combined_df = combined_df[
            ['Sr No.', 'Dataset_Split', 'Dialogue_ID', 'Utterance_ID', 'Season', 'Episode', 'StartTime', 'EndTime',
             'Utterance', 'Speaker', 'Sentiment', 'Emotion', 'Emotion_Label', 'TextEmbeddings', 'SpeechEmbeddings',
             'MultimodalEmbeddings']]
        return combined_df  # .drop(['TextEmbeddings', 'SpeechEmbeddings'], axis=1)

    def create_multimodal_embeddings(self, output_file):
        df = self.get_multimodal_embeddings()
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    speech_vectors_file_ = 'data/meld_xvectors.csv'
    text_vectors_files_ = {
        'dev': 'data/dev_sent_emo_norm_word_embedded.csv',
        'train': 'data/train_sent_emo_norm_word_embedded.csv',
        'test': 'data/test_sent_emo_norm_word_embedded.csv',
    }
    generator = MultiModalEmbeddingGeneratorMeld(speech_vectors_file_, text_vectors_files_)
    generator.create_multimodal_embeddings('data/dataset_with_multi_modal_embeddings.csv')
    print('ok')
