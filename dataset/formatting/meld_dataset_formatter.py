import pandas as pd

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


EMOTION_IDS = {
    'anger/disgust': 0,
    'fear/surprise': 1,
    'happiness': 2,
    'neutral': 3,
    'sadness': 4
}

MELD_EMOTION_MAP = {
    'joy': 'happiness',
    'sadness': 'sadness',
    'surprise': 'fear/surprise',
    'fear': 'fear/surprise',
    'anger': 'anger/disgust',
    'disgust': 'anger/disgust',
    'neutral': 'neutral'
}


class MeldFormatter:
    """
    This class is used to format the MELD CSVs to normalize the emotion labels and add an ordinal emotion column
    """
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.special_characters = {
            chr(133): '...',
            chr(145): '\'',
            chr(146): '\'',
            chr(147): '"',
            chr(148): '"',
            chr(150): '-',
            chr(151): '-',
            chr(160): ' ',
            chr(233): 'e',
            chr(243): 'o',
            chr(8212): '-',
            chr(8217): '\'',
            chr(8230): '...',
        }

    def normalize_unicode_characters(self, string):
        for character, replacement in self.special_characters.items():
            string = string.replace(character, replacement)
        return string

    def format_emotion_labels(self):
        self.data = self.data.assign(Emotion_Label='no')
        self.data['Utterance'] = self.data['Utterance'].apply(self.normalize_unicode_characters)
        for emotion, normalized_emotion in MELD_EMOTION_MAP.items():
            self.data.loc[self.data['Emotion'] == emotion, 'Emotion_Label'] = EMOTION_IDS[normalized_emotion]

    def write_formatted_data(self, file_name):
        self.data.to_csv(file_name, index=False)


if __name__ == '__main__':
    for split in ['dev', 'test', 'train']:
        source_csv = 'data/' + split + '_sent_emo.csv'
        target_csv = source_csv.replace('_emo', '_emo_norm_2')
        formatter = MeldFormatter(source_csv)
        formatter.format_emotion_labels()
        formatter.write_formatted_data(target_csv)
