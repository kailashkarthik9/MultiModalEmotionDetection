#!/usr/bin/env python

# author: aa4461

import csv
import os.path
import sys

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'

COLLAPSED_IEMOCAP_EMOTIONS = {
    '0': 'anger/disgust',
    '1': 'fear/surprise',
    '2': 'happiness',
    '3': 'neutral',
    '4': 'sadness'
}


class UtteranceDetails:
    def __init__(self, session, mocap_source, dialogue_type,
                 dialogue_number, utterance_number, utterance, speaker,
                 src_file, orig_emotion, mapped_emotion):
        self.session = session
        self.mocap_source = mocap_source
        self.dialogue_type = dialogue_type
        self.dialogue_number = dialogue_number
        self.utterance_number = utterance_number
        self.utterance = utterance
        self.speaker = speaker
        self.src_file = src_file
        self.orig_emotion = orig_emotion
        self.mapped_emotion = mapped_emotion

    def __repr__(self):
        return "%s: %s" % (self.get_filename(), self.utterance)

    # old id format (might be required for nnet scoring)
    def get_old_id(self):
        return "%s_Ses%s%s_%s%s_%s%s" % (
            self.orig_emotion,
            self.session,
            self.mocap_source,
            self.get_dialogue_type_shortname(),
            self.dialogue_number,
            self.speaker,
            self.utterance_number
        )

    # new id format (required for LDA / PLDA training)
    def get_id(self):
        return "%s-%s-%s%s-%s-%s-%s-%s" % (
            self.mapped_emotion, self.src_file, self.session, self.mocap_source,
            self.dialogue_type, self.dialogue_number, self.utterance_number, self.speaker)

    def get_dialogue_type_shortname(self):
        if self.dialogue_type == "improvisation":
            return "impro"
        elif self.dialogue_type == "script":
            return "script"
        else:
            raise Exception("Unsupported dialogue type: %s" % (self.dialogue_type))

    # example filename: Ses01F_impro01/Ses01F_impro01_F000.wav
    def get_filename(self):
        return "Ses%s%s_%s%s/%s.wav" % (
            self.session,
            self.mocap_source,
            self.get_dialogue_type_shortname(),
            self.dialogue_number,
            self.get_old_id()[2:]
        )


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
                # example CSV row
                # 0 Sr.No: 1303
                # 1 Session_Number: 01
                # 2 Mocap_Source: F
                # 3 Dialogue_Type: improvisation
                # 4 Dialogue_Number: 01
                # 5 Utterance_Number: 000
                # 6 StartTime: 6.2901
                # 7 EndTime: 8.2357
                # 8 Utterance: Excuse me.
                # 9 Speaker: F
                # 10 Emotion: neu
                # 11 Emotion_Label: 3
                session = row[1]
                mocap_source = row[2]
                dialogue_type = row[3]
                dialogue_number = row[4]
                utterance_number = row[5]
                utterance = row[8]
                speaker = row[9]
                orig_emotion = row[11]
                mapped_emotion = COLLAPSED_IEMOCAP_EMOTIONS[orig_emotion]
                utterances.append(
                    UtteranceDetails(
                        session,
                        mocap_source,
                        dialogue_type,
                        dialogue_number,
                        utterance_number,
                        utterance,
                        speaker,
                        src_file,
                        orig_emotion,
                        mapped_emotion
                    )
                )
                line_count += 1
    return utterances


# note that in the utt2spk file we generate we map each
# utterance to a numerical encoding of the utterance's 
# emotion label according to IEMOCAP (doesn't matter thaaat much)
def generate_utt2spk(utterances, output_data_dir):
    with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
        for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
            f.write("%s %s\n" % (utterance.get_id(), utterance.orig_emotion))


def generate_wavscp(utterances, input_data_dir, output_data_dir):
    with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
        for utterance in sorted(utterances, key=lambda utterance: utterance.get_id()):
            wav_file_path = os.path.join(input_data_dir, utterance.get_filename())
            if not os.path.exists(wav_file_path):
                raise Exception('Missing file: %s' % (utterance.get_filename()))
            wav_creation_cmd = "ffmpeg -v 8 -i %s -f wav -ar 16000 -acodec pcm_s16le -|" % (wav_file_path)
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
