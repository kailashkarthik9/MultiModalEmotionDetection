#!/usr/bin/env python

# author: aa4461

import sys

import kaldi_io


def get_predictions(prediction_filepaths):
    utt2predictions = {}
    for prediction_filepath in prediction_filepaths:
        for utterance_id, predictions in kaldi_io.read_mat_ark(prediction_filepath):
            if utterance_id in utt2predictions:
                raise Exception('%s in multiple files!' % (utterance_id))
            utt2predictions[utterance_id] = predictions
    return utt2predictions


def main():
    prediction_filepaths = sys.argv[1:-1]
    output_filepath = sys.argv[-1]

    all_predictions = get_predictions(prediction_filepaths)
    with open(output_filepath, 'wb') as f:
        for utterance_id, prediction in all_predictions.iteritems():
            kaldi_io.write_mat(f, prediction, key=utterance_id)


if __name__ == "__main__":
    main()
