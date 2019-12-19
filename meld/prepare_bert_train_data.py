import pandas as pd

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


class DataPreparer:
    def __init__(self, dev_data, train_data, test_data):
        self.dev_df = pd.read_csv(dev_data)
        self.train_df = pd.read_csv(train_data)
        self.test_df = pd.read_csv(test_data)

    def prepare_lm_data(self):
        utterances = self.dev_df['Utterance'].tolist() + self.train_df['Utterance'].tolist() + self.test_df[
            'Utterance'].tolist()
        utterances_text = ' '.join(utterances)
        weird_characters = set()
        for char_ in utterances_text:
            if ord(char_) > 127:
                weird_characters.add(ord(char_))
        return utterances

    def create_lm_data(self, file_name):
        prepared_data = self.prepare_lm_data()
        lm_file = open(file_name, 'w')
        lm_file.write('\n'.join(prepared_data))
        lm_file.close()


if __name__ == '__main__':
    dev_csv = 'data/dev_sent_emo_norm.csv'
    train_csv = 'data/train_sent_emo_norm.csv'
    test_csv = 'data/test_sent_emo_norm.csv'
    output_txt = 'data/utterances_text.txt'
    preparer = DataPreparer(dev_csv, train_csv, test_csv)
    preparer.prepare_lm_data()
    preparer.create_lm_data(output_txt)
