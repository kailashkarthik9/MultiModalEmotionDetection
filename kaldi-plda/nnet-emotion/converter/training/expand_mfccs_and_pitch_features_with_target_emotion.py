#!/usr/bin/env python

# author: aa4461

import glob
import os.path
import sys

import kaldiio
import numpy as np


def get_target_emotion(utt):
    return int(utt[0])


def write_expanded_feature(raw_mfcc_and_pitch_file, output_data_dir):
    expanded_features = {}
    for utt, features in kaldiio.load_ark(raw_mfcc_and_pitch_file):
        num_frames = len(features)
        target_emotion_column = np.full((num_frames, 1), get_target_emotion(utt))
        expanded_feature = np.append(features, target_emotion_column, 1)
        expanded_features[utt] = expanded_feature

    (_, split, _) = raw_mfcc_and_pitch_file.split('.', 2)
    kaldiio.save_ark(
        os.path.join(output_data_dir, 'mfcc_pitch_and_target_emotion.%s.ark' % (split)),
        expanded_features,
        scp=os.path.join(output_data_dir, 'mfcc_pitch_and_target_emotion.%s.scp' % (split))
    )


def main():
    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[2]

    for raw_mfcc_and_pitch_file in glob.glob('%s/*ark' % (input_data_dir)):
        write_expanded_feature(raw_mfcc_and_pitch_file, output_data_dir)


if __name__ == "__main__":
    main()
