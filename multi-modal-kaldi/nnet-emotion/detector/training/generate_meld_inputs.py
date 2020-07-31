#!/usr/bin/env python

# author: aa4461

import csv
import os.path
import sys

COLLAPSED_MELD_EMOTIONS = {
    'anger': 'anger/disgust',
    'disgust': 'anger/disgust',
    'fear': 'fear/surprise',
    'joy': 'happiness',
    'neutral': 'neutral',
    'sadness': 'sadness',
    'surprise': 'fear/surprise'
}
# note that changing this set of emotions (via addition, removal or reordering)
# requires re-generating the MELD input data and re-training the final model!
EMOTION_TO_ID = {emotion: id for id, emotion in
                 enumerate(['anger/disgust', 'fear/surprise', 'happiness', 'neutral', 'sadness'])}

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'


class UtteranceDetails:
    def __init__(self, src_file, dialogue_id, utterance_id, emotion):
        self.src_file = src_file
        self.dialogue_id = dialogue_id
        self.utterance_id = utterance_id
        self.emotion = emotion

    def __repr__(self):
        return "%s: %s" % (self.get_id(), self.emotion)

    def get_id(self):
        return "%s-%s-%s-%s" % (
            self.emotion, self.src_file, self.dialogue_id, self.utterance_id)

    def get_filename(self):
        return "dia%s_utt%s.mp4" % (self.dialogue_id, self.utterance_id)


def get_utterances(input_csv):
    src_file = None
    if input_csv.find("train") != -1:
        src_file = "train"
    elif input_csv.find("dev") != -1:
        src_file = "dev"
    else:
        src_file = "test"

    utterances = []
    with open(input_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                meld_emotion = row[3]
                mapped_emotion = COLLAPSED_MELD_EMOTIONS[meld_emotion]
                utterances.append(UtteranceDetails(src_file, row[5], row[6], mapped_emotion))
                line_count += 1
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
            mp4_file_path = os.path.join(input_data_dir, utterance.get_filename())
            wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (mp4_file_path)
            f.write("%s %s\n" % (utterance.get_id(), wav_creation_cmd))


def main():
    input_csv = sys.argv[1]
    input_data_dir = sys.argv[2]
    output_data_dir = sys.argv[3]

    utterances = get_utterances(input_csv)

    generate_utt2spk(utterances, output_data_dir)
    generate_wavscp(utterances, input_data_dir, output_data_dir)


if __name__ == "__main__":
    main()
