#!/usr/bin/env python

# author: aa4461

import os.path
import sys

import scoring_utils

NUM_EMOTIONS = 5

UTT2SPK_FILE = 'utt2spk'
WAV_FILE = 'wav.scp'
PREDICTIONS_FILE = 'predictions.ark'


def get_wav_labels_and_predictions(input_dir, prefix):
    utts_to_wav_creation = {}
    with open(os.path.join(input_dir, '%s_%s' % (prefix, WAV_FILE)), 'r') as f:
        for line in f:
            (utt, wav_creation) = line.split(None, 1)
            if utt in utts_to_wav_creation:
                raise Exception('%s duped in %s wav file' % (utt, prefix))
            utts_to_wav_creation[utt] = wav_creation.strip()

    utts_to_labels = {}
    with open(os.path.join(input_dir, '%s_%s' % (prefix, UTT2SPK_FILE)), 'r') as f:
        for line in f:
            (utt, label) = line.split(None, 1)
            if utt in utts_to_labels:
                raise Exception('%s duped in %s utt2spk file' % (utt, prefix))
            utts_to_labels[utt] = int(label)

    utts_to_predictions = scoring_utils.parse_predictions_ark(
        os.path.join(input_dir, '%s_%s' % (prefix, PREDICTIONS_FILE)))

    return (utts_to_wav_creation, utts_to_labels, utts_to_predictions)


def get_correctly_predicted_utterances(utt_to_labels, utt_to_predictions):
    correctly_predicted = []
    for utt in utt_to_predictions.keys():
        label = utt_to_labels[utt]
        prediction = scoring_utils.get_most_likely_emotion(utt_to_predictions[utt])
        if label == prediction:
            correctly_predicted.append(utt)
    return correctly_predicted


def get_training_examples(delimiter, utts, utts_to_wav_creation):
    training_examples = []
    for utt in utts:
        _, base_utt = utt.split(delimiter, 1)
        for emotion in range(0, NUM_EMOTIONS):
            training_examples.append(('%s%s%s' % (emotion, delimiter, base_utt), emotion, utts_to_wav_creation[utt]))
    return training_examples


def generate_utt2spk(utterances, output_data_dir):
    with open(os.path.join(output_data_dir, UTT2SPK_FILE), 'w') as f:
        for utterance in sorted(utterances, key=lambda utterance: utterance[0]):
            f.write("%s %s\n" % (utterance[0], utterance[1]))


def generate_wavscp(utterances, output_data_dir):
    with open(os.path.join(output_data_dir, WAV_FILE), 'w') as f:
        for utterance in sorted(utterances, key=lambda utterance: utterance[0]):
            f.write("%s %s\n" % (utterance[0], utterance[2]))


def main():
    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]

    meld_wav, meld_labels, meld_predictions = get_wav_labels_and_predictions(input_data_dir, 'meld')
    iemocap_wav, iemocap_labels, iemocap_predictions = get_wav_labels_and_predictions(input_data_dir, 'iemocap')

    meld_correctly_predicted = get_correctly_predicted_utterances(meld_labels, meld_predictions)
    iemocap_correctly_predicted = get_correctly_predicted_utterances(iemocap_labels, iemocap_predictions)

    print('Generating a training corpus using %s correctly predicted utterances (%s from MELD, %s from IEMOCAP)'
          % (len(meld_correctly_predicted) + len(iemocap_correctly_predicted), len(meld_correctly_predicted),
             len(iemocap_correctly_predicted)))

    training_examples = get_training_examples('-', meld_correctly_predicted, meld_wav) + get_training_examples('_',
                                                                                                               iemocap_correctly_predicted,
                                                                                                               iemocap_wav)

    generate_utt2spk(training_examples, output_data_dir)
    generate_wavscp(training_examples, output_data_dir)


if __name__ == "__main__":
    main()
