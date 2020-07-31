#!/usr/bin/env python

# author: aa4461

import csv
import glob
import os.path
import sys

import numpy as np

COLLAPSED_CREMA_D_EMOTIONS = {
    'A': 'anger/disgust',
    'D': 'anger/disgust',
    'F': 'fear/surprise',
    'H': 'happiness',
    'N': 'neutral',
    'S': 'sadness'
}
# note that changing this set of emotions (via addition, removal or reordering)
# requires re-generating the CremaD input data and re-training the final model!
EMOTION_TO_ID = {emotion: id for id, emotion in
                 enumerate(['anger/disgust', 'fear/surprise', 'happiness', 'neutral', 'sadness'])}

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'


class UtteranceDetails:
    def __init__(self, file_name, file_path, emotion):
        self.file_name = file_name
        self.file_path = file_path
        self.emotion = emotion

    def __repr__(self):
        return "%s: %s" % (self.file_name, self.emotion)

    def get_id(self):
        return "%s-%s" % (self.emotion, self.file_name)


# A, D, F, H, N, or S
# "","FileName","VoiceVote","VoiceLevel","FaceVote","FaceLevel","MultiModalVote","MultiModalLevel"
def get_utterances(label_type, vote_threshold, input_csv, wav_filepaths):
    if label_type == 'voice':
        label_col_prefix = 'Voice'
    else:
        label_col_prefix = 'MultiModal'

    utterances = []
    with open(input_csv) as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for row in csv_reader:
            file_name = row["FileName"]
            wav_filepath = wav_filepaths[file_name]
            voice_votes = row["%sVote" % label_col_prefix].split(":")
            voice_levels = list(map(lambda l: float(l), row["%sLevel" % label_col_prefix].split(":")))

            if vote_threshold == 'all' and len(voice_votes) > 1:
                continue

            crema_d_emotion = voice_votes[np.argmax(voice_levels)]
            mapped_emotion = COLLAPSED_CREMA_D_EMOTIONS[crema_d_emotion]
            utterances.append(UtteranceDetails(file_name, wav_filepath, mapped_emotion))
    return utterances


# note that the utt2spk file we generate actually each
# utterance to a numerical encoding of the utterance's 
# emotion label according to MELD (so that we can lever 
# the existing vox2/run.sh with minimal modification)
def generate_utt2spk(utterances, output_data_dir):
    with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
        for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
            f.write("%s %s\n" % (utterance.get_id(), EMOTION_TO_ID[utterance.emotion]))


def generate_wavscp(utterances, input_data_dir, output_data_dir):
    with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
        for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
            f.write("%s %s\n" % (utterance.get_id(), utterance.file_path))


INPUT_CSV = 'processedResults/summaryTable.csv'
WAV_DIR = 'AudioWAV'


def main():
    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]
    cremad_label_type = sys.argv[3]
    cremad_vote_threshold = sys.argv[4]

    wav_filepaths = {w.split('/')[-1].split('.')[0]: w \
                     for w in glob.glob(os.path.join(input_data_dir, WAV_DIR, '*.wav'))}
    utterances = get_utterances(
        cremad_label_type, cremad_vote_threshold, os.path.join(input_data_dir, INPUT_CSV), wav_filepaths)

    generate_utt2spk(utterances, output_data_dir)
    generate_wavscp(utterances, input_data_dir, output_data_dir)


if __name__ == "__main__":
    main()
